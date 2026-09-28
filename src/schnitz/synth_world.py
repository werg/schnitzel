"""A seeded fictional world of people and companies as a redundancy-controlled knowledge task
(after Allen-Zhu and Li, Physics of Language Models 3.1, bioS with multiM / permute
augmentation, and PhantomWiki's generated universes).

Purpose (owner, 28 September): a first L1 read task whose answer comes from the KB and whose
redundancy is a controlled knob. ``World(seed, people)`` draws people (unique natural full
names, gender, birth date and city, university, major, employer, job title, start year, an
optional mentor and sibling) and companies (city, industry, founding year) from built-in
vocabularies. ``records(world, redundancy)`` states every fact in exactly ``redundancy``
records, counted across record types:

- ``bio``: ``redundancy`` biography paragraphs per person, each from its own random
  templates (about 50 phrasings per attribute: a subject form - full name, pronoun, first
  name, title - times a predicate template), sentence order permuted, the first sentence
  always with the full name;
- ``roster``: company staff lists (employee, job title, start year), one line per employee;
- ``directory``: city birth registers (person, birth date), one line per person born there;
- ``alumni``: university alumni lists (person, major);
- ``company``: company profiles (city, industry, founding year), ``redundancy`` per company.

Each list line is one copy of its facts, so a person's bios state an attribute
``redundancy - (list copies)`` times; a sibling pair's relation is split over both siblings'
bios. Every record's provenance lists the facts it states (``facts``: ``"<pid>:<attribute>"``
or ``"c<cid>:<attribute>"``), which is what episodes use to name every copy.

``questions(world, facts_index, ...)`` asks about one attribute of one person (1-hop) or,
with ``hops=2``, the city/industry of a person's employer or an attribute of their mentor.
Train and validation are split by person: validation people's records are in the KB but no
training question asks about them.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import random

DOMAIN = 'synth_people'
PROMPT = 'Use the stored records. Give only the short answer.\nQuestion: '
RECORD_CHARS = 1500
MONTHS = ('January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
          'September', 'October', 'November', 'December')

FEMALE = '''Abigail Ada Adele Agnes Alice Alma Amelia Anna Aurora Beatrice Bella Bianca Brenda
Bridget Camille Carla Caroline Cecilia Charlotte Chloe Clara Claudia Cora Daisy Daphne Delia
Diana Dora Edith Eleanor Elena Eliza Ella Eloise Elsie Emma Esther Eva Evelyn Fiona Flora
Frances Freya Gemma Georgia Grace Greta Hannah Harriet Hazel Helen Ida Imogen Iris Isabel Ivy
Jane Joanna Josephine Julia June Katherine Laura Leah Lena Lillian Lucy Lydia Mabel Madeline
Margaret Maria Marian Martha Matilda Maya Mildred Miriam Nadia Naomi Nina Nora Olive Olivia
Paula Pearl Penelope Phoebe Priscilla Rachel Rebecca Rosa Rose Ruby Ruth Sabrina Sarah Selma
Sophia Stella Susanna Tabitha Tessa Thea Vera Victoria Violet Vivian Wilma Yvonne Zara Zoe'''.split()
MALE = '''Aaron Adrian Albert Alexander Alfred Ambrose Andrew Anthony Arthur August Barnaby
Benedict Benjamin Bernard Caleb Calvin Carl Casper Cedric Charles Christopher Clement Conrad
Cyrus Daniel David Declan Desmond Dominic Duncan Edgar Edmund Edward Elias Elliot Emil Ernest
Ezra Felix Ferdinand Francis Frederick Gabriel George Gideon Gilbert Gordon Graham Gregory Harold
Harvey Henry Herbert Hugo Isaac Ivan Jacob Jasper Jonah Joseph Julian Kenneth Laurence Leo
Leonard Lewis Lionel Louis Lucas Magnus Malcolm Marcus Martin Matthew Maxwell Miles Nathan
Neville Nicholas Oliver Oscar Otto Patrick Percy Peter Philip Quentin Ralph Raymond Robert
Roland Rupert Samuel Sebastian Silas Simon Stanley Theodore Thomas Tobias Victor Vincent Walter
Wesley William Xavier'''.split()
LAST = '''Abbott Ainsworth Aldridge Ashby Atwood Bancroft Barlow Beckett Bellamy Blackwood
Bradshaw Brennan Brightwell Calloway Carrington Chandler Chatterton Crawford Cromwell Dalton
Davenport Delacroix Drummond Dunmore Easton Ellery Everhart Fairbanks Falkner Fenwick Fletcher
Forsythe Galloway Garrick Gilmore Goodwin Granger Greaves Hadley Halloway Hargrove Hartley
Hastings Hawthorne Holloway Huxley Ingram Irving Jarvis Kendrick Kingsley Kirkland Lancaster
Langley Latimer Lockwood Lowell Lyndon Mallory Marlowe Merriweather Montague Morrow Northcott
Norwood Oakley Ogilvie Pemberton Pennington Prescott Quinlan Radcliffe Ramsey Redfield Rowntree
Rutherford Sallow Sanderson Sinclair Southgate Stanhope Sterling Stratton Sutcliffe
Talbot Thackeray Thornton Tilford Townsend Underwood Vance Vickers Wadsworth Wakefield Walcott
Warrington Westbrook Whitaker Whitmore Wilder Winslow Woodward Wraxall Yardley Yates Zeller'''.split()
CITIES = '''Aberdeen Adelaide Albany Amsterdam Antwerp Athens Atlanta Austin Baltimore Barcelona
Bergen Bern Bilbao Birmingham Bologna Bordeaux Boston Bremen Brisbane Bristol Brussels
Budapest Calgary Cardiff Charleston Chicago Cincinnati Cleveland Copenhagen Cork Dallas Denver
Detroit Dublin Dundee Edinburgh Florence Frankfurt Geneva Genoa Ghent Glasgow Gothenburg Graz
Halifax Hamburg Hanover Helsinki Honolulu Houston Indianapolis Innsbruck Kyoto Leeds Leipzig
Lille Lisbon Liverpool Ljubljana Lyon Madrid Manchester Marseille Melbourne Memphis Milan
Milwaukee Montreal Munich Nantes Naples Nashville Newcastle Nottingham Oakland Omaha Oslo Ottawa
Oxford Palermo Perth Philadelphia Phoenix Pittsburgh Porto Portland Prague Quebec Raleigh
Reykjavik Richmond Riga Rotterdam Sacramento Salzburg Seattle Seville Sheffield Stockholm
Strasbourg Stuttgart Sydney Tallinn Tampere Toronto Toulouse Trieste Turin Utrecht Valencia
Vancouver Venice Vienna Vilnius Warsaw Wellington Winnipeg York Zagreb Zurich'''.split()
UNI_PLACES = '''Ashbury Bramwell Carrowmore Dunstan Eastmere Fairhaven Glenwood Harrowgate
Ironbridge Juniper Kestrel Larkspur Marbury Northfield Oakridge Pellham Queensgate Ravenhill
Silverbrook Thornbury Upton Valemont Westmoor Whitcombe Yarrow Aldermere Birchfield Coldharbour
Dovecote Elmstead'''.split()
UNI_FORMS = ('{p} University', 'University of {p}', '{p} Institute of Technology', '{p} College',
             '{p} State University')
MAJORS = '''Accounting|Anthropology|Applied Mathematics|Architecture|Art History|Astronomy|
Biochemistry|Biology|Chemical Engineering|Chemistry|Civil Engineering|Classics|Computer Science|
Economics|Education|Electrical Engineering|English Literature|Environmental Science|Finance|
Geography|Geology|History|Industrial Design|Journalism|Linguistics|Marine Biology|Marketing|
Materials Science|Mathematics|Mechanical Engineering|Music|Neuroscience|Nursing|Philosophy|
Physics|Political Science|Psychology|Public Health|Sociology|Statistics|Urban Planning|
Veterinary Science'''.replace('\n', '').split('|')
INDUSTRIES = '''aerospace|agriculture|biotechnology|construction|consulting|cybersecurity|
education technology|energy|financial services|food processing|furniture|insurance|logistics|
machine tools|media|medical devices|mining|pharmaceuticals|publishing|real estate|renewable energy|
retail|robotics|semiconductors|shipping|software|telecommunications|textiles|tourism|
water treatment'''.replace('\n', '').split('|')
CO_HEAD = '''Amberline Bluewater Brightforge Cedarpoint Clearpath Copperleaf Crestview Deepfield
Driftwood Eastgate Emberstone Fernhill Flintlock Goldcrest Granite Harborlight Highmark Ironwood
Kingfisher Lakeshore Lanternfield Meridian Northstar Oakhaven Pinecrest Quarrystone Redwood
Riverbend Saltmarsh Silverline Stonebridge Summit Thistle Timberline Truewind Westfield
Whitestone Willowmere Yellowpine Zephyr'''.split()
CO_TAIL = ('Systems', 'Analytics', 'Industries', 'Labs', 'Partners', 'Holdings', 'Dynamics',
           'Works', 'Group', 'Technologies', 'Logistics', 'Solutions')
TITLES = '''accountant|analyst|architect|chief engineer|data scientist|designer|economist|
field engineer|financial controller|hardware engineer|laboratory technician|lawyer|
logistics coordinator|marketing manager|mechanical engineer|operations manager|product manager|
project manager|quality inspector|recruiter|research scientist|sales director|senior analyst|
software engineer|systems administrator|technical writer|test engineer|translator|
procurement officer|chief financial officer'''.replace('\n', '').split('|')
HOBBIES = ('birdwatching', 'chess', 'rowing', 'pottery', 'amateur astronomy', 'sailing',
           'woodworking', 'choral singing', 'rock climbing', 'watercolour painting',
           'beekeeping', 'fencing', 'cycling', 'bookbinding', 'orienteering', 'the cello')

ATTRIBUTES = ('birth_date', 'birth_city', 'university', 'major', 'employer', 'job_title',
              'start_year', 'mentor', 'sibling')
COMPANY_ATTRIBUTES = ('company_city', 'industry', 'founded')
# list records that state each person attribute (one copy per list line)
LIST_COPIES = {'birth_date': 1, 'birth_city': 1, 'university': 1, 'major': 1, 'employer': 1,
               'job_title': 1, 'start_year': 1, 'mentor': 0, 'sibling': 0}

# predicate templates; {s} is the subject form, {S} its capitalized form
PREDICATES = {
    'birth_date': ['{S} was born on {v}.', '{S} came into the world on {v}.',
                   'The birth date of {s} is {v}.', '{S} celebrates a birthday each year, '
                   'having been born on {v}.', 'Records give {v} as the birth date of {s}.',
                   '{S} was born on {v}, a fact noted in every official file.',
                   'On {v}, {s} was born.', '{S} entered life on {v}.',
                   'The day {s} was born was {v}.', '{S} has a birth date of {v}.'],
    'birth_city': ['{S} was born in {v}.', '{S} is a native of {v}.', '{S} hails from {v}.',
                   'The birthplace of {s} is {v}.', '{S} grew up in {v}, where {p} was born.',
                   '{v} is the city where {s} was born.', '{S} spent early childhood in '
                   '{v}, the city of birth.', '{S} was born and raised in {v}.',
                   'Born in {v}, {s} still speaks fondly of the city.',
                   'The city of {v} is where {s} first saw the light of day.'],
    'university': ['{S} studied at {v}.', '{S} graduated from {v}.', '{S} attended {v}.',
                   '{S} earned a degree at {v}.', '{S} is a graduate of {v}.',
                   'For university, {s} went to {v}.', '{v} is where {s} completed a degree.',
                   '{S} received a diploma from {v}.', '{S} enrolled at {v} after school.',
                   'The alma mater of {s} is {v}.'],
    'major': ['{S} majored in {v}.', '{S} studied {v} at university.',
              '{S} holds a degree in {v}.', 'The field {s} chose at university was {v}.',
              '{S} specialised in {v} as a student.', '{S} earned a degree in {v}.',
              'At university, {s} focused on {v}.', '{v} was the major of {s}.',
              '{S} completed a course of study in {v}.', 'The academic major of {s} was {v}.'],
    'employer': ['{S} works for {v}.', '{S} is employed by {v}.', '{S} has a job at {v}.',
                 '{v} employs {s}.', '{S} joined {v} and still works there.',
                 'The employer of {s} is {v}.', '{S} earns a living at {v}.',
                 'Professionally, {s} is with {v}.', '{S} is on the staff of {v}.',
                 '{S} spends working days at {v}.'],
    'job_title': ['{S} works as {a} {v}.', 'By profession, {s} is {a} {v}.',
                  '{S} holds the position of {v}.', 'The job title of {s} is {v}.',
                  '{S} serves as {a} {v}.', '{S} is employed as {a} {v}.',
                  'In the office, {s} is known as {a} {v}.', '{S} has the role of {v}.',
                  'The current role of {s} is {v}.', '{S} earns a salary as {a} {v}.'],
    'start_year': ['{S} started the current job in {v}.', '{S} joined the employer in {v}.',
                   'Since {v}, {s} has held the current position.',
                   '{S} began working at the company in {v}.',
                   'The year {s} was hired was {v}.', '{S} was hired in {v}.',
                   '{S} took up the present post in {v}.',
                   'In {v}, {s} started at the current employer.',
                   '{S} has been with the current employer since {v}.',
                   'Company records show that {s} started in {v}.'],
    'mentor': ['{S} was mentored by {v}.', 'The mentor of {s} is {v}.',
               '{v} served as mentor to {s}.', '{S} learned the trade from {v}, a mentor.',
               '{S} credits {v} as mentor.', 'Early in the career of {s}, {v} was the mentor.',
               '{S} was guided by the mentor {v}.', '{v} mentored {s} for several years.',
               'As a mentor, {v} shaped the career of {s}.', '{S} names {v} as mentor.'],
    'sibling': ['{S} has a {rel}, {v}.', '{S} is the {inv} of {v}.',
                '{v} is the {rel} of {s}.', '{S} grew up with a {rel} named {v}.',
                'The {rel} of {s} is {v}.', '{S} and {v} are siblings.',
                '{S} often visits {v}, who is a {rel}.',
                '{S} shares a family home at holidays with {v}, a {rel}.',
                'In the family, {s} has one sibling: {v}.', '{v} and {s} are siblings.'],
    'hobby': ['In spare time, {s} enjoys {v}.', '{S} is fond of {v}.',
              'A favourite pastime of {s} is {v}.', '{S} has long been devoted to {v}.'],
}
COMPANY_TEMPLATES = {
    'company_city': ['{c} is headquartered in {v}.', '{c} is based in {v}.',
                     'The head office of {c} is in {v}.', '{c} has its headquarters in {v}.'],
    'industry': ['{c} operates in the {v} industry.', '{c} is a {v} company.',
                 'The business of {c} is {v}.', '{c} works in {v}.'],
    'founded': ['{c} was founded in {v}.', '{c} was established in {v}.',
                'The founding year of {c} is {v}.', '{c} has existed since {v}.'],
}
QUESTIONS = {
    'birth_date': ['When was {n} born?', 'What is the birth date of {n}?',
                   'On what date was {n} born?', 'What is the date of birth of {n}?'],
    'birth_city': ['Where was {n} born?', 'In which city was {n} born?',
                   'What is the birthplace of {n}?', 'Which city is {n} a native of?'],
    'university': ['Which university did {n} attend?', 'Where did {n} study?',
                   'From which university did {n} graduate?', 'What is the alma mater of {n}?'],
    'major': ['What did {n} study?', 'What was the major of {n}?',
              'In which field does {n} hold a degree?', 'What subject did {n} major in?'],
    'employer': ['Which company does {n} work for?', 'Who employs {n}?',
                 'What is the employer of {n}?', 'Where does {n} work?'],
    'job_title': ['What is the job title of {n}?', 'What does {n} work as?',
                  'What position does {n} hold?', 'What is the role of {n} at work?'],
    'start_year': ['In which year did {n} start the current job?',
                   'When did {n} join the current employer?',
                   'Since which year has {n} worked at the current employer?',
                   'What year was {n} hired by the current employer?'],
    'mentor': ['Who mentored {n}?', 'Who is the mentor of {n}?',
               'Who served as mentor to {n}?', 'Whom does {n} name as mentor?'],
    'sibling': ['Who is the sibling of {n}?', 'What is the name of the sibling of {n}?',
                'Who is the brother or sister of {n}?', 'Which person is a sibling of {n}?'],
}
TWO_HOP = {  # (first hop attribute, second hop attribute): question templates
    ('employer', 'company_city'): ['In which city is the employer of {n} headquartered?',
                                   'Where is the company that employs {n} based?'],
    ('employer', 'industry'): ['In which industry does the employer of {n} operate?',
                               'What business is the company {n} works for in?'],
    ('mentor', 'birth_city'): ['Where was the mentor of {n} born?',
                               'In which city was the person who mentored {n} born?'],
    ('mentor', 'employer'): ['Which company does the mentor of {n} work for?',
                             'Who employs the mentor of {n}?'],
}


def record_id(domain: str, body: str) -> str:
    return hashlib.sha256(f'{domain}\0{body}'.encode()).hexdigest()[:32]


def _rng(seed: int, *parts) -> random.Random:
    key = '\0'.join(str(p) for p in (seed, *parts))
    return random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))


@dataclass
class Company:
    cid: int
    name: str
    city: str
    industry: str
    founded: int

    def value(self, attribute: str) -> str:
        return str({'company_city': self.city, 'industry': self.industry,
                    'founded': self.founded}[attribute])


@dataclass
class Person:
    pid: int
    first: str
    middle: str
    last: str
    female: bool
    birth: tuple[int, int, int]           # year, month, day
    birth_city: str
    university: str
    major: str
    employer: int
    job_title: str
    start_year: int
    hobby: str
    mentor: int | None = None
    sibling: int | None = None

    @property
    def name(self) -> str:
        return f'{self.first} {self.middle} {self.last}'

    @property
    def birth_date(self) -> str:
        y, m, d = self.birth
        return f'{MONTHS[m - 1]} {d}, {y}'


@dataclass
class World:
    seed: int
    people: list[Person] = field(default_factory=list)
    companies: list[Company] = field(default_factory=list)

    def value(self, pid: int, attribute: str) -> str | None:
        p = self.people[pid]
        if attribute == 'birth_date':
            return p.birth_date
        if attribute == 'employer':
            return self.companies[p.employer].name
        if attribute in ('mentor', 'sibling'):
            other = getattr(p, attribute)
            return None if other is None else self.people[other].name
        return str(getattr(p, attribute))


def make_world(seed: int, people: int = 20000, *, mentor_rate: float = 0.8,
               sibling_rate: float = 0.3) -> World:
    """Deterministic in ``seed``; full names are unique within the world."""
    rng = _rng(seed, 'world')
    universities = sorted({form.format(p=place) for place in UNI_PLACES for form in UNI_FORMS})
    universities = rng.sample(universities, min(len(universities), max(20, people // 40)))
    n_companies = max(12, people // 25)
    names = rng.sample([f'{h} {t}' for h in CO_HEAD for t in CO_TAIL],
                       min(n_companies, len(CO_HEAD) * len(CO_TAIL)))
    world = World(seed)
    for cid, name in enumerate(names):
        world.companies.append(Company(cid, name, rng.choice(CITIES), rng.choice(INDUSTRIES),
                                       rng.randint(1890, 2012)))
    used: set[str] = set()
    families: list[int] = []
    while len(world.people) < people:
        pid = len(world.people)
        female = rng.random() < 0.5
        pool = FEMALE if female else MALE
        sibling_of = families.pop() if families and rng.random() < 0.5 else None
        last = world.people[sibling_of].last if sibling_of is not None else rng.choice(LAST)
        first, middle = rng.choice(pool), rng.choice(pool)
        full = f'{first} {middle} {last}'
        if first == middle or full in used:
            if sibling_of is not None:
                families.append(sibling_of)
            continue
        used.add(full)
        year = rng.randint(1950, 2000)
        company = world.companies[rng.randrange(len(world.companies))]
        start = rng.randint(max(year + 22, company.founded), 2025)
        world.people.append(Person(
            pid, first, middle, last, female, (year, rng.randint(1, 12), rng.randint(1, 28)),
            rng.choice(CITIES), rng.choice(universities), rng.choice(MAJORS), company.cid,
            rng.choice(TITLES), start, rng.choice(HOBBIES)))
        if sibling_of is not None:
            world.people[pid].sibling = sibling_of
            world.people[sibling_of].sibling = pid
        elif rng.random() < sibling_rate:
            families.append(pid)
    for p in world.people:
        if rng.random() < mentor_rate:
            other = rng.randrange(len(world.people))
            if other != p.pid and other != p.sibling:
                p.mentor = other
    return world


# -- records -------------------------------------------------------------------------
def _record(text: str, record_type: str, facts: list[str], **provenance) -> dict:
    text = text.strip()
    return {'record_id': record_id(DOMAIN, text), 'text': text, 'domain': DOMAIN,
            'created_at': 1, 'kind': 'passage',
            'provenance': {'dataset': DOMAIN, 'record_type': record_type,
                           'facts': sorted(set(facts)), **provenance}}


def _pack(header: str, lines: list[tuple[str, list[str]]], limit: int = RECORD_CHARS):
    """Lines (text, facts) into texts of at most ``limit`` characters under ``header``."""
    out, current, facts = [], [], []
    for line, line_facts in lines:
        if current and len(header) + sum(len(x) + 1 for x in current) + len(line) > limit:
            out.append((header + '\n'.join(current), facts))
            current, facts = [], []
        current.append(line)
        facts = facts + line_facts
    if current:
        out.append((header + '\n'.join(current), facts))
    return out


def _subject(p: Person, rng: random.Random, first_sentence: bool, pronoun_rate: float):
    if first_sentence:
        return p.name
    r = rng.random()
    if r < pronoun_rate:
        return 'she' if p.female else 'he'
    if r < pronoun_rate + 0.15:
        return p.first
    if r < pronoun_rate + 0.25:
        return f'{"Ms." if p.female else "Mr."} {p.last}'
    return p.name


def _sentence(world: World, p: Person, attribute: str, rng: random.Random, first: bool,
              pronoun_rate: float) -> str:
    template = rng.choice(PREDICATES[attribute])
    subject = _subject(p, rng, first, pronoun_rate)
    if template.startswith('{v}') or '{S}' not in template:
        # the subject is mid-sentence: a pronoun would be ungrammatical as possessive
        subject = p.name if subject in ('she', 'he') else subject
    value = p.hobby if attribute == 'hobby' else world.value(p.pid, attribute)
    extra = {'a': 'an' if str(value)[0] in 'aeiou' else 'a'}
    if attribute == 'sibling':
        other = world.people[p.sibling]
        extra.update(rel='sister' if other.female else 'brother',
                     inv='sister' if p.female else 'brother')
    text = template.format(s=subject, S=subject[0].upper() + subject[1:], v=value,
                           p='she' if p.female else 'he', **extra)
    return text[0].upper() + text[1:]


def _unique(make, seen: set[str], tries: int = 200) -> str:
    for _ in range(tries):
        text = make()
        if text not in seen:
            seen.add(text)
            return text
    raise ValueError('cannot draw another distinct copy; lower the redundancy')


def bio_plan(world: World, redundancy: int, rng: random.Random) -> dict[int, list[set[str]]]:
    """Per person, the attribute set of each of its ``redundancy`` bios, so that every
    fact is stated ``redundancy`` times across bios and list records."""
    plan = {p.pid: [set() for _ in range(redundancy)] for p in world.people}
    for p in world.people:
        for attribute in ATTRIBUTES:
            if world.value(p.pid, attribute) is None or attribute == 'sibling':
                continue
            need = max(0, redundancy - LIST_COPIES[attribute])
            for i in rng.sample(range(redundancy), need):
                plan[p.pid][i].add(attribute)
        if p.sibling is not None and p.pid < p.sibling:
            # the pair's relation: split over both siblings' bios
            mine = (redundancy + 1) // 2
            for i in rng.sample(range(redundancy), mine):
                plan[p.pid][i].add('sibling')
            for i in rng.sample(range(redundancy), redundancy - mine):
                plan[p.sibling][i].add('sibling')
    return plan


def records(world: World, redundancy: int = 8, *, pronoun_rate: float = 0.5,
            hobby_rate: float = 0.5) -> list[dict]:
    """Every record of the world (see the module docstring), deterministic in the seed."""
    if redundancy < 1:
        raise ValueError('redundancy must be at least 1')
    rng = _rng(world.seed, 'records', redundancy)
    out: list[dict] = []
    seen: set[str] = set()     # copies are distinct texts (a record id is its text's hash)
    plan = bio_plan(world, redundancy, rng)
    for p in world.people:
        for i, attributes in enumerate(plan[p.pid]):
            if not attributes:
                continue

            def bio(attributes=attributes):
                order = list(attributes)
                rng.shuffle(order)
                if rng.random() < hobby_rate:
                    order.insert(rng.randint(1, len(order)), 'hobby')
                return ' '.join(_sentence(world, p, a, rng, k == 0, pronoun_rate)
                                for k, a in enumerate(order))
            text = _unique(bio, seen)
            facts = [f'{p.pid}:{a}' for a in attributes if a != 'sibling']
            if 'sibling' in attributes:
                facts += [f'{p.pid}:sibling', f'{p.sibling}:sibling']
            out.append(_record(text, 'bio', facts, person=p.pid, copy=i))
    # company rosters, profiles
    staff: dict[int, list[Person]] = {}
    for p in world.people:
        staff.setdefault(p.employer, []).append(p)
    for c in world.companies:
        lines = [(f'- {p.name}, {p.job_title}, since {p.start_year}',
                  [f'{p.pid}:employer', f'{p.pid}:job_title', f'{p.pid}:start_year'])
                 for p in sorted(staff.get(c.cid, []), key=lambda p: (p.last, p.first, p.pid))]
        for part, (text, facts) in enumerate(_pack(f'Staff roster of {c.name} (employee, job '
                                                    'title, start year):\n', lines)):
            out.append(_record(text, 'roster', facts, company=c.cid, part=part))
        for i in range(redundancy):

            def profile(c=c):
                order = list(COMPANY_ATTRIBUTES)
                rng.shuffle(order)
                return ' '.join(rng.choice(COMPANY_TEMPLATES[a]).format(c=c.name, v=c.value(a))
                                for a in order)
            text = _unique(profile, seen)
            out.append(_record(text, 'company', [f'c{c.cid}:{a}' for a in COMPANY_ATTRIBUTES],
                               company=c.cid, copy=i))
    # city birth registers and alumni lists
    groups: dict[tuple[str, str], list[Person]] = {}
    for p in world.people:
        groups.setdefault(('directory', p.birth_city), []).append(p)
        groups.setdefault(('alumni', p.university), []).append(p)
    for (kind, key), members in sorted(groups.items()):
        members = sorted(members, key=lambda p: (p.last, p.first, p.pid))
        if kind == 'directory':
            header = f'Register of people born in {key} (name, date of birth):\n'
            lines = [(f'- {p.name}, born {p.birth_date}',
                      [f'{p.pid}:birth_city', f'{p.pid}:birth_date']) for p in members]
        else:
            header = f'Alumni list of {key} (graduate, major):\n'
            lines = [(f'- {p.name}, {p.major}', [f'{p.pid}:university', f'{p.pid}:major'])
                     for p in members]
        for part, (text, facts) in enumerate(_pack(header, lines)):
            out.append(_record(text, kind, facts, group=key, part=part))
    return out


def fact_index(recs: list[dict]) -> dict[str, list[str]]:
    """fact -> ids of the records that state it, in record order."""
    index: dict[str, list[str]] = {}
    for rec in recs:
        for fact in rec['provenance']['facts']:
            index.setdefault(fact, []).append(rec['record_id'])
    return index


def copy_distribution(recs: list[dict], facts: list[str] | None = None) -> dict:
    """Per-fact copy statistics: histogram of copies per fact and of the record-type mix."""
    types = {r['record_id']: r['provenance']['record_type'] for r in recs}
    index = fact_index(recs)
    wanted = facts if facts is not None else list(index)
    copies = Counter(len(index.get(f, [])) for f in wanted)
    mix = Counter(' + '.join(f'{t}:{n}' for t, n in sorted(Counter(
        types[r] for r in index.get(f, [])).items())) for f in wanted)
    return {'facts': len(wanted), 'copies_per_fact': dict(sorted(copies.items())),
            'record_type_mix': dict(mix.most_common(12))}


# -- episodes ------------------------------------------------------------------------
def split_of(world: World, pid: int, validation: float) -> str:
    return 'validation' if _rng(world.seed, 'split', pid).random() < validation else 'train'


def _episode(identifier: str, split: str, question: str, answer: str, per_hop: list[list[str]],
             by_id: dict[str, dict], rng: random.Random, family: str, max_groups: int,
             **provenance) -> dict:
    """Every record of every hop is a support; each single record (1 hop) or each sampled
    combination of one record per hop is a sufficient group. ``alternatives`` keeps the
    records per hop so a transcript slot can name all copies of its fact."""
    hops = [list(h) for h in per_hop]
    for h in hops:
        rng.shuffle(h)
    if len(hops) == 1:
        groups = [[r] for r in hops[0]]
    else:
        groups = []
        seen = set()
        for _ in range(max_groups * 4):
            g = tuple(rng.choice(h) for h in hops)
            if g not in seen:
                seen.add(g)
                groups.append(list(g))
            if len(groups) >= max_groups:
                break
    everything = list(dict.fromkeys(r for h in hops for r in h))
    return {
        'episode_id': f'{DOMAIN}-{identifier}', 'environment': f'{DOMAIN}-{split}',
        'query': PROMPT + question, 'answer': answer, 'query_time': 2,
        'required_ids': everything, 'sufficient_groups': groups,
        'alternatives': hops, 'support_annotation': 'verified', 'task_family': family,
        'supports': [{'record_id': r, 'text': by_id[r]['text'], 'created_at': 1,
                      'kind': by_id[r]['kind']} for r in everything],
        'verify': {'type': 'exact', 'answer': answer},
        'provenance': {'dataset': DOMAIN, 'domain': DOMAIN, 'split': split, **provenance},
    }


def questions(world: World, recs: list[dict], *, validation: float = 0.1, hops: int = 1,
              two_hop_rate: float = 0.3, max_groups: int = 32) -> dict[str, list[dict]]:
    """Episodes per split: every stated attribute of every person (1-hop), plus with
    ``hops=2`` a ``two_hop_rate`` share of people with one 2-hop question each."""
    by_id = {r['record_id']: r for r in recs}
    index = fact_index(recs)
    out: dict[str, list[dict]] = {'train': [], 'validation': []}
    for p in world.people:
        split = split_of(world, p.pid, validation)
        rng = _rng(world.seed, 'questions', p.pid)
        for attribute in ATTRIBUTES:
            answer = world.value(p.pid, attribute)
            if answer is None:
                continue
            question = rng.choice(QUESTIONS[attribute]).format(n=p.name)
            out[split].append(_episode(
                f'{p.pid}-{attribute}', split, question, answer, [index[f'{p.pid}:{attribute}']],
                by_id, rng, 'synthetic_people_qa', max_groups, person=p.pid,
                attribute=attribute, hops=1, world_seed=world.seed))
        if hops >= 2 and rng.random() < two_hop_rate:
            options = [k for k in TWO_HOP if getattr(p, k[0]) is not None]
            first, second = options[rng.randrange(len(options))]
            if first == 'employer':
                mid, key = p.employer, f'c{p.employer}:{second}'
                answer = world.companies[mid].value(second)
            else:
                mid = p.mentor
                key, answer = f'{mid}:{second}', world.value(mid, second)
            question = rng.choice(TWO_HOP[first, second]).format(n=p.name)
            out[split].append(_episode(
                f'{p.pid}-{first}-{second}', split, question, answer,
                [index[f'{p.pid}:{first}'], index[key]], by_id, rng, 'synthetic_people_multihop',
                max_groups, person=p.pid, attribute=f'{first}.{second}', hops=2,
                world_seed=world.seed))
    return out


def build(seed: int = 0, people: int = 20000, redundancy: int = 8, *, hops: int = 1,
          validation: float = 0.1, two_hop_rate: float = 0.3, max_groups: int = 32):
    """World, records, episodes per split and a summary (copies per asked fact by type)."""
    world = make_world(seed, people)
    recs = records(world, redundancy)
    episodes = questions(world, recs, validation=validation, hops=hops,
                         two_hop_rate=two_hop_rate, max_groups=max_groups)
    asked = sorted({f'{e["provenance"]["person"]}:{e["provenance"]["attribute"]}'
                    for rows in episodes.values() for e in rows if e['provenance']['hops'] == 1})
    sizes = [len(r['text']) for r in recs]
    summary = {'seed': seed, 'people': people, 'companies': len(world.companies),
               'redundancy': redundancy, 'hops': hops, 'validation': validation,
               'records_by_type': dict(Counter(r['provenance']['record_type'] for r in recs)),
               'record_chars': {'mean': round(sum(sizes) / len(sizes)), 'max': max(sizes)},
               'asked_fact_copies': copy_distribution(recs, asked)}
    return world, recs, episodes, summary
