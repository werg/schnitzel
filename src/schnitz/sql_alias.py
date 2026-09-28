"""Schema aliasing for text-to-SQL task corpora (``scripts/prepare_task_corpora.py``).

A database variant renames the database id, every table and every column to fresh
identifiers, so the names a query must use are not guessable from the question and
have to come from the knowledge base. One namespace per database: every distinct
lower-cased table or column name maps to one fresh identifier (a column name shared
by several tables keeps one alias, so join keys stay recognisable), which makes the
rewrite context free and exactly invertible.

- ``alias_names`` draws the fresh names from a seeded RNG: ``semantic`` (default)
  keeps them meaningful but not guessable (synonyms, abbreviations, a different
  naming convention, table prefixes, plural changes, reordered compounds; one
  convention per variant), ``random`` uses unrelated word pairs and codes. Names are
  unique within the database, never equal to an original name (also not modulo
  case and separators) and never SQL keywords or common function names.
- ``rename_sql`` rewrites SQL with a tokenizer: string literals (``'...'``) and
  comments are kept; quoted identifiers (``"x"``, `` `x` ``, ``[x]``) and bare words
  are matched case-insensitively (``alias.column`` too, part by part); a bare word
  followed by ``(`` is a function and a bare structural keyword is never renamed.
  ``strict`` raises ``AliasError`` for ambiguous rewrites: a double-quoted token that
  names a schema object right after a comparison operator, ``LIKE`` or inside ``IN
  (...)`` (SQLite may read it as a string literal), and a query alias (``AS w``,
  ``FROM t w``) equal to a schema name.
- ``alias_query`` renames gold SQL and checks that the inverse rename gives the
  original back (modulo identifier case outside literals).
- ``rename_create`` rewrites a ``CREATE TABLE`` statement positionally (the table,
  each column definition's name, ``REFERENCES`` targets and identifiers inside
  nested parentheses; type words are kept).
- ``rename_text`` rewrites identifier mentions in notes and evidence: identifier-like
  names (with ``_``, a digit or internal capitals) at word boundaries, other names
  only in identifier syntax (quoted, dotted or followed by a comparison operator), so
  descriptions keep their meaning.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import random
import re
import unicodedata

TOKEN = re.compile(r"""
    (?P<str>'(?:[^']|'')*')
  | (?P<comment>--[^\n]*|/\*.*?\*/)
  | (?P<dq>"(?:[^"]|"")*")
  | (?P<bt>`[^`]*`)
  | (?P<br>\[[^\]]*\])
  | (?P<num>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\w*)
  | (?P<word>[^\W\d]\w*)
  | (?P<ws>\s+)
  | (?P<op><=|>=|<>|!=|==|\|\||.)
""", re.S | re.X)

# words that are never renamed when bare (structural SQL); a schema name among them is
# used quoted in valid SQL
STRUCTURAL = frozenset('''select from where and or not in is null like glob between group by
order having limit offset union intersect except all distinct case when then else end as on
join inner left right outer cross natural using exists cast asc desc collate escape with
insert update delete create table into values set primary foreign key references default
constraint unique check index if replace true false'''.split())
# every SQLite keyword and common function name: never a fresh name
SQL_WORDS = STRUCTURAL | frozenset('''
abort action add after all alter always analyze and as asc attach autoincrement before
begin between by cascade case cast check collate column commit conflict constraint
create cross current current_date current_time current_timestamp database default
deferrable deferred delete desc detach distinct do drop each else end escape except
exclude exclusive exists explain fail filter first following for foreign from full
generated glob group groups having if ignore immediate in index indexed initially inner
insert instead intersect into is isnull join key last left like limit match materialized
natural no not nothing notnull null nulls of offset on or order others outer over
partition plan pragma preceding primary query raise range recursive references regexp
reindex release rename replace restrict returning right rollback row rows savepoint
select set table temp temporary then ties to transaction trigger unbounded union unique
update using vacuum values view virtual when where window with without count sum avg min
max total abs round length lower upper substr substring trim ltrim rtrim instr date time
datetime julianday strftime coalesce ifnull nullif iif printf typeof random real integer
int text blob numeric float double varchar char boolean bigint smallint decimal
timestamp rowid oid main true false'''.split())
HEAD_WORDS = frozenset('create table if not exists temp temporary'.split())
COMPARE = frozenset({'=', '==', '!=', '<>', '<', '>', '<=', '>=', 'like', 'glob'})
IDENT = re.compile(r'[A-Za-z_][A-Za-z0-9_]*\Z')


class AliasError(ValueError):
    """An ambiguous rewrite; ``reason`` names it (counted as a corpus filter)."""

    def __init__(self, reason: str, detail: str = ''):
        super().__init__(f'{reason}: {detail}' if detail else reason)
        self.reason = reason


def tokenize(sql: str) -> list[tuple[str, str]]:
    return [(m.lastgroup, m.group()) for m in TOKEN.finditer(sql)]


def _inner(kind: str, text: str) -> str:
    if kind == 'dq':
        return text[1:-1].replace('""', '"')
    return text[1:-1]


def _quote(kind: str, name: str) -> str:
    return {'dq': f'"{name}"', 'bt': f'`{name}`', 'br': f'[{name}]'}[kind]


def _significant(tokens) -> list[int]:
    return [i for i, (k, _) in enumerate(tokens) if k not in ('ws', 'comment')]


def rename_sql(sql: str, mapping: Mapping[str, str], strict: bool = True) -> str:
    """``sql`` with every identifier in ``mapping`` (lower-case name -> new name)
    replaced (module docstring); ``strict`` raises ``AliasError`` for ambiguity."""
    tokens = tokenize(sql)
    sig = _significant(tokens)
    pos = {i: n for n, i in enumerate(sig)}
    out = [t for _, t in tokens]

    def at(n: int) -> tuple[str, str]:
        return tokens[sig[n]] if 0 <= n < len(sig) else ('', '')

    def low(n: int) -> str:
        return at(n)[1].lower()

    depth = 0
    parens: list[bool] = []            # per open paren: opened right after IN
    from_at: set[int] = set()          # depths whose FROM clause is open
    expect_table = expect_alias = False
    for i, (kind, text) in enumerate(tokens):
        if kind in ('ws', 'comment', 'str', 'num'):
            if kind in ('str', 'num'):
                expect_table = expect_alias = False
            continue
        n = pos[i]
        prev, nxt = low(n - 1), low(n + 1)
        if kind == 'op':
            if text == '(':
                parens.append(prev == 'in')
                depth += 1
                expect_table = expect_alias = False
            elif text == ')':
                if parens:
                    parens.pop()
                from_at.discard(depth)
                depth = max(0, depth - 1)
                expect_table = expect_alias = False
            elif text == ',':
                expect_table, expect_alias = depth in from_at, False
            else:
                expect_table = expect_alias = False
            continue
        lower = text.lower() if kind == 'word' else _inner(kind, text).lower()
        # a schema name that is a keyword (a table ``cast``) used bare where only a name
        # can stand: in table position or before ``.``
        keyword_name = (kind == 'word' and lower in STRUCTURAL and lower in mapping
                        and nxt != '(' and (expect_table or nxt == '.'))
        if kind == 'word' and lower in STRUCTURAL and not keyword_name:
            if lower == 'from':
                from_at.add(depth)
            elif lower in ('where', 'group', 'order', 'having', 'limit', 'union', 'intersect',
                           'except', 'on', 'using'):
                from_at.discard(depth)
            expect_table = lower in ('from', 'join')
            expect_alias = False
            continue
        is_alias = prev == 'as' or (expect_alias and kind == 'word')
        if is_alias and strict and lower in mapping and nxt != '(':
            raise AliasError('alias_collision', text)
        if expect_table:
            expect_table, expect_alias = False, True
        else:
            expect_alias = False
        if is_alias or lower not in mapping:
            continue
        if kind == 'word':
            if nxt == '(':
                continue                       # a function call
            out[i] = mapping[lower]
            continue
        if kind == 'dq' and strict and (prev in COMPARE or
                                        (prev in ('(', ',') and parens and parens[-1])):
            raise AliasError('quoted_literal_ambiguous', text)
        out[i] = _quote(kind, mapping[lower])
    return ''.join(out)


def normalized(sql: str) -> list[str]:
    """Significant tokens with identifiers lower-cased (literals kept as written)."""
    return [t if k in ('str', 'num') else (t.lower() if k in ('word', 'op') else
                                           k + ':' + _inner(k, t).lower())
            for k, t in tokenize(sql) if k not in ('ws', 'comment')]


def inverse_of(mapping: Mapping[str, str], originals: Mapping[str, str]) -> dict[str, str]:
    """new name (lower case) -> original spelling; ``originals`` maps lower -> spelling."""
    return {new.lower(): originals.get(old, old) for old, new in mapping.items()}


def alias_query(sql: str, mapping: Mapping[str, str], inverse: Mapping[str, str]) -> str:
    """The renamed gold SQL, checked to invert exactly (``AliasError`` otherwise)."""
    renamed = rename_sql(sql, mapping)
    try:
        back = rename_sql(renamed, inverse)
    except AliasError as error:
        raise AliasError('alias_roundtrip', str(error)) from None
    if normalized(back) != normalized(sql):
        raise AliasError('alias_roundtrip', sql)
    return renamed


def unmapped_originals(sql: str, inverse: Mapping[str, str]) -> set[str]:
    """Original schema names used as identifiers in ``sql`` that are not fresh names
    (an aliased answer that guessed the original schema)."""
    originals = {v.lower() for v in inverse.values()} - set(inverse)
    found = set()
    tokens = tokenize(sql)
    sig = _significant(tokens)
    for n, i in enumerate(sig):
        kind, text = tokens[i]
        if kind == 'word':
            nxt = tokens[sig[n + 1]][1] if n + 1 < len(sig) else ''
            if nxt != '(' and text.lower() not in STRUCTURAL and text.lower() in originals:
                found.add(text.lower())
        elif kind in ('dq', 'bt', 'br') and _inner(kind, text).lower() in originals:
            found.add(_inner(kind, text).lower())
    return found


def rename_create(create: str, mapping: Mapping[str, str]) -> str:
    """A ``CREATE TABLE`` statement renamed positionally (module docstring); comments
    get ``rename_text``."""
    tokens = tokenize(create)
    out = [t for _, t in tokens]
    depth, start, after_ref = 0, False, False

    def put(i: int, kind: str, text: str) -> None:
        lower = text.lower() if kind == 'word' else _inner(kind, text).lower()
        if lower in mapping:
            out[i] = mapping[lower] if kind == 'word' else _quote(kind, mapping[lower])

    sig = _significant(tokens)
    nxt = {i: tokens[sig[n + 1]][1] if n + 1 < len(sig) else '' for n, i in enumerate(sig)}
    for i, (kind, text) in enumerate(tokens):
        if kind == 'comment':
            out[i] = rename_text(text, mapping)
            continue
        if kind in ('ws', 'str', 'num'):
            continue
        if kind == 'op':
            if text == '(':
                depth += 1
                start = depth == 1
            elif text == ')':
                depth -= 1
            elif text == ',' and depth == 1:
                start, after_ref = True, False
            continue
        if depth == 0:
            ident = kind != 'word' or text.lower() not in HEAD_WORDS
        else:
            ident = kind != 'word' or text.lower() not in STRUCTURAL
        if depth == 0:
            if ident:
                put(i, kind, text)
        elif depth == 1:
            if start:        # a column definition's name, or a table constraint
                start = False
                if not (kind == 'word' and text.lower() in ('constraint', 'primary', 'foreign',
                                                            'unique', 'check')):
                    put(i, kind, text)
                continue
            if after_ref:
                after_ref = False
                put(i, kind, text)
            elif kind == 'word' and text.lower() == 'references':
                after_ref = True
        elif ident and nxt.get(i) != '(':
            put(i, kind, text)
    return ''.join(out)


QUOTED = r'`([^`]+)`|"([^"]+)"|\[([^\]]+)\]'
TEXT_TOKEN = re.compile(rf"{QUOTED}|'[^'\n]{{0,200}}'|(\w+(?:\.\w+)+)|(\w+)(\s*(?:[=<>]|!=))?")


def identifier_like(name: str) -> bool:
    return bool(re.fullmatch(r'\w+', name)) and bool(
        '_' in name or re.search(r'\d', name) or re.search(r'[a-z][A-Z]|[A-Z]{2}', name))


def rename_text(text: str, mapping: Mapping[str, str]) -> str:
    """Identifier mentions in free text renamed (module docstring)."""
    def sub(m: re.Match) -> str:
        whole = m.group(0)
        for g, (left, right) in zip((1, 2, 3), (('`', '`'), ('"', '"'), ('[', ']'))):
            if m.group(g) is not None:
                new = mapping.get(m.group(g).lower())
                return f'{left}{new}{right}' if new else whole
        if m.group(4) is not None:
            return '.'.join(mapping.get(p.lower(), p) for p in whole.split('.'))
        word = m.group(5)
        if word is None:
            return whole                            # a single-quoted value
        new = mapping.get(word.lower())
        if new and (identifier_like(word) or m.group(6) is not None):
            return new + (m.group(6) or '')
        return whole
    return TEXT_TOKEN.sub(sub, text)


# -- fresh names ----------------------------------------------------------------------
# a small hand table of schema-word synonyms (no downloads); every entry maps a word to
# alternatives that keep its meaning
SYNONYMS = {w: alts.split() for w, alts in (line.split(':') for line in '''
id:ident key num code ref
name:label title designation moniker
title:heading label caption
date:day dated when_on
year:yr annum season
time:moment clock hour
age:years_old oldness
city:town municipality locale
country:nation land state_name
state:province region_name territory
address:addr location street
phone:telephone tel contact_number
email:mail e_mail
price:cost charge rate
cost:expense price outlay
amount:sum_value quantity figure
total:overall aggregate combined
number:num count_of figure
type:kind category sort
status:state_flag condition standing
code:tag ref_code identifier
description:details summary info
rank:position standing placing
score:points mark tally
rating:grade stars assessment
level:tier grade stage
grade:mark level score_band
student:pupil learner scholar
teacher:instructor educator tutor
course:subject module class_unit
class:section cohort
school:academy institute college
department:dept division unit
employee:staff worker personnel
manager:supervisor boss head
salary:pay wage earnings
company:firm business corporation
customer:client buyer patron
order:purchase booking request
product:item goods article
item:article entry piece
store:shop outlet retailer
shop:store outlet boutique
sale:sold deal transaction
payment:remittance settlement pay
invoice:bill receipt
account:acct ledger profile
bank:lender institution
loan:credit advance borrowing
movie:film picture feature
film:movie picture feature
director:filmmaker helmer directed_by
actor:performer cast_member player
show:program broadcast
song:track tune piece
singer:vocalist performer artist
artist:creator performer maker
album:record release
music:audio tunes
track:song cut
genre:style category
book:volume title_work publication
author:writer creator
publisher:press imprint
player:athlete competitor
team:squad side club
game:match contest fixture
match:game fixture bout
season:campaign year_span
league:division competition
club:society association
coach:trainer manager
stadium:arena venue ground
event:occasion happening
race:contest heat
driver:racer pilot
car:vehicle auto
airport:airfield aerodrome
flight:journey sortie
airline:carrier
aircraft:plane airplane
ship:vessel boat
train:rail locomotive
station:stop terminal depot
route:path way
hotel:inn lodging
room:chamber suite
apartment:flat unit
building:structure edifice
member:participant affiliate
user:account_holder member
person:individual human
people:persons individuals
visitor:guest caller
museum:gallery exhibit_hall
hospital:clinic infirmary
doctor:physician medic
patient:case client
category:class group
color:colour hue
size:dimension magnitude
weight:mass heft
height:stature elevation
length:extent span
population:inhabitants residents headcount
area:region zone surface
region:area zone territory
district:borough ward zone
county:shire parish
location:place site spot
latitude:lat
longitude:lon lng
start:begin opening onset
end:finish close stop
first:initial earliest
last:final latest
gender:sex
birth:born
death:died passing
nationality:citizenship
language:tongue lang
party:faction
election:vote poll ballot
candidate:nominee contender
budget:allocation funds
revenue:income turnover earnings
profit:gain margin
market:marketplace exchange
industry:sector trade
owner:holder proprietor
staff:personnel crew
festival:fest celebration
concert:gig performance
capacity:max_occupancy volume
document:doc paper file
card:pass
transaction:trans transfer
policy:plan cover
claim:demand request
enrollment:enrolment registration
note:remark memo
comment:remark feedback
review:critique evaluation
detail:particular specific
record:entry log_entry
history:past log
version:revision release
issue:problem topic
task:job assignment
project:initiative scheme
role:function part
position:post place
job:role occupation
contract:agreement deal
result:outcome finding
disease:illness condition
treatment:therapy care
molecule:compound
atom:element_unit
element:component
label:tag marker
power:strength ability
hero:champion protagonist
circuit:track course
constructor:builder maker
lap:circuit_round loop
term:period tenure
medal:award prize
athlete:sportsperson competitor
sport:discipline
post:message entry
vote:ballot tally
tag:label keyword
character:figure role
chapter:section part
word:term token
page:leaf sheet
beer:brew ale
business:enterprise firm
inspection:audit check_up
violation:breach offence
trip:journey ride
weather:climate conditions
zip:postcode postal
app:application program
client:customer patron
free:complimentary no_cost
meal:food lunch
count:tally quantity
'''.strip().splitlines())}
ABBREVIATIONS = {'number': 'no', 'identifier': 'id', 'department': 'dept', 'account': 'acct',
                 'address': 'addr', 'quantity': 'qty', 'amount': 'amt', 'average': 'avg_',
                 'description': 'descr', 'information': 'info', 'category': 'cat',
                 'director': 'dir', 'name': 'nm', 'date': 'dt', 'count': 'cnt',
                 'employee': 'emp', 'customer': 'cust', 'product': 'prod', 'student': 'stu',
                 'country': 'ctry', 'population': 'pop', 'reference': 'ref', 'total': 'tot',
                 'minimum': 'min_', 'maximum': 'max_', 'percent': 'pct', 'percentage': 'pct',
                 'year': 'yr', 'month': 'mo', 'value': 'val', 'status': 'stat',
                 'location': 'loc', 'organization': 'org', 'transaction': 'txn',
                 'message': 'msg', 'temperature': 'temp_', 'school': 'sch', 'grade': 'grd'}
PREFIXES = ('', '', 'tbl_', 't_', 'tb_')
CONVENTIONS = ('snake', 'snake', 'camel', 'pascal')
# unrelated words for ``random`` names
POOL = sorted({w for alts in SYNONYMS.values() for a in alts for w in a.split('_')
               if len(w) >= 3 and w not in SQL_WORDS} | {
    'amber', 'birch', 'cobalt', 'delta', 'ember', 'fjord', 'garnet', 'harbor', 'indigo',
    'juniper', 'kestrel', 'lagoon', 'maple', 'nectar', 'onyx', 'pepper', 'quartz', 'raven',
    'saffron', 'tundra', 'umber', 'velvet', 'walnut', 'yarrow', 'zephyr'})


def words_of(name: str) -> list[str]:
    """Lower-case ASCII words of a name (separators, camelCase and digit boundaries)."""
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode()
    name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', name)
    return [w.lower() for w in re.findall(r'[A-Za-z]+|\d+', name)]


def _squash(name: str) -> str:
    return re.sub(r'[^a-z0-9]', '', name.lower())


def _plural(word: str) -> str:
    if word.endswith('ies'):
        return word[:-3] + 'y'
    if word.endswith('s') and not word.endswith('ss'):
        return word[:-1]
    if word.endswith('y') and len(word) > 2 and word[-2] not in 'aeiou':
        return word[:-1] + 'ies'
    return word + ('es' if word.endswith(('s', 'x', 'ch', 'sh')) else 's')


def _abbreviate(word: str) -> str:
    if word in ABBREVIATIONS:
        return ABBREVIATIONS[word].rstrip('_')
    if len(word) <= 4 or word.isdigit():
        return word
    squeezed = word[0] + re.sub(r'[aeiou]', '', word[1:])
    return squeezed[:5] if len(squeezed) >= 3 else word[:4]


class _Variant:
    """The seeded choices of one variant: convention, table prefix, plural toggle and
    one replacement per word (so a word reads the same in every name of the variant)."""

    def __init__(self, rng: random.Random, style: str):
        self.rng, self.style = rng, style
        self.convention = rng.choice(CONVENTIONS)
        self.prefix = rng.choice(PREFIXES)
        self.plural = rng.random() < 0.5
        self.words: dict[str, str] = {}

    def word(self, w: str, force: bool = False) -> str:
        if w in self.words and not force:
            return self.words[w]
        rng, options = self.rng, []
        if w in SYNONYMS:
            options += SYNONYMS[w] * 3
        if not w.isdigit() and len(w) > 4:
            options.append(_abbreviate(w))
        if not force:
            options.append(w)
        if not options:
            options = [w + rng.choice(('x', 'v', '_val', '_ref'))]
        choice = rng.choice(options)
        if not force:
            self.words[w] = choice
        return choice

    def join(self, parts: list[str]) -> str:
        parts = [p for part in parts for p in part.split('_') if p]
        if self.convention == 'snake':
            name = '_'.join(parts)
        else:
            name = ''.join(p if i == 0 and self.convention == 'camel' else p[:1].upper() + p[1:]
                           for i, p in enumerate(parts))
        return name if re.match(r'[A-Za-z_]', name) else 'n_' + name

    def name(self, original: str, table: bool, attempt: int) -> str:
        rng = self.rng
        if self.style == 'random':
            name = rng.choice(POOL) + '_' + rng.choice(POOL)
            return name if attempt < 3 else f'{name}_{rng.randrange(10, 99)}'
        words = words_of(original) or ['field']
        parts = [self.word(w, force=attempt > 0 and rng.random() < 0.6) for w in words]
        if attempt > 0 and all(p == w for p, w in zip(parts, words)):
            k = rng.randrange(len(parts))
            parts[k] = self.word(words[k], force=True)
        if len(parts) > 1 and rng.random() < 0.25:
            parts = parts[1:] + parts[:1]
        if table and self.plural:
            parts[-1] = _plural(parts[-1]) if not parts[-1].isdigit() else parts[-1]
        if attempt >= 3:
            parts.append(rng.choice(('rec', 'ref', 'info', 'set', 'log', 'item')))
        if attempt >= 6:
            parts.append(str(rng.randrange(2, 99)))
        return (self.prefix if table else '') + self.join(parts)


def _valid(name: str, taken: set[str], originals: set[str]) -> bool:
    return bool(IDENT.match(name)) and name.lower() not in SQL_WORDS \
        and name.lower() not in taken and _squash(name) not in originals and len(name) <= 48


def alias_names(tables: Mapping[str, Iterable[str]], rng: random.Random,
                style: str = 'semantic', reserved: Iterable[str] = ()) -> dict[str, str]:
    """Fresh names for every table and column (lower-case original -> new name), one
    namespace per database (module docstring); ``reserved`` names are not used."""
    if style not in ('semantic', 'random'):
        raise ValueError(f'alias style is semantic or random, not {style!r}')
    variant = _Variant(rng, style)
    spelled: dict[str, tuple[str, bool]] = {}
    for table, columns in tables.items():
        if table.lower().startswith('sqlite_'):
            continue                  # SQLite's own tables (sqlite_sequence) keep their names
        spelled.setdefault(table.lower(), (table, True))
        for column in columns:
            spelled.setdefault(column.lower(), (column, False))
    originals = {_squash(n) for n in spelled} | {_squash(n) for n in reserved}
    taken = {n.lower() for n in reserved}
    out = {}
    for lower in sorted(spelled):
        original, table = spelled[lower]
        for attempt in range(40):
            name = variant.name(original, table, attempt)
            if _valid(name, taken, originals):
                break
        else:
            raise RuntimeError(f'no fresh name for {original!r}')
        out[lower] = name
        taken.add(name.lower())
    return out


def alias_db(db_id: str, rng: random.Random, style: str, handle: str,
             taken: set[str], originals: Iterable[str] = ()) -> str:
    """A database handle for one variant, unique among ``taken`` (updated): the
    database id aliased like a table name (``name``) or an opaque code (``opaque``)."""
    squashed = {_squash(o) for o in originals} | {_squash(db_id)}
    for attempt in range(60):
        if handle == 'opaque':
            name = f'db_{rng.getrandbits(24):06x}'
        else:
            variant = _Variant(rng, style)
            variant.prefix = ''
            name = variant.name(db_id, False, attempt)
        if _valid(name, taken, squashed):
            taken.add(name.lower())
            return name
    raise RuntimeError(f'no fresh database handle for {db_id!r}')
