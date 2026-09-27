from dataclasses import replace

import pytest
import torch

from schnitz.evaluation_interventions import TargetedPayloadStore
from schnitz.store import DiskStore, StoredRecord, ReadPlan, Selection


def fixture(tmp_path):
    base, changed = DiskStore(tmp_path/'base.sqlite'), DiskStore(tmp_path/'changed.sqlite')
    for i in range(2):
        record = StoredRecord(str(i), torch.ones(2), torch.full((3,),float(i)), source_id=str(i), created_at=1)
        base.put(record)
        changed.put(replace(record,payload=record.payload+10))
    plan = ReadPlan('default','s0','v0','research',10,(Selection('0',0.),Selection('1',0.)))
    return base,changed,plan


def test_only_selected_intervention_targets_use_alternative_values(tmp_path):
    base,changed,plan = fixture(tmp_path)
    store = TargetedPayloadStore(base,changed,frozenset({'0'}))
    values = store.fetch(plan)
    torch.testing.assert_close(values[0],torch.full((3,),10.))
    torch.testing.assert_close(values[1],torch.ones(3))
    # An unselected target cannot affect any consumed value.
    untouched = TargetedPayloadStore(base,changed,frozenset({'unselected'})).fetch(plan)
    for a,b in zip(untouched,base.fetch(plan)):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    with pytest.raises(ValueError, match='fixed'):
        store.search(torch.ones(2))


@pytest.mark.parametrize('change', ['base_deleted','changed_deleted','domain','time','source','shape','dtype'])
def test_intervention_preserves_both_stores_visibility_and_provenance(tmp_path,change):
    base,changed,plan = fixture(tmp_path)
    if change.endswith('deleted'):
        (base if change=='base_deleted' else changed).delete('default','0')
    else:
        column,value={'domain':('domain','private'),'time':('created_at',10),'source':('source_id','other'),
                      'shape':('payload',None),'dtype':('payload',None)}[change]
        if value is None:
            from safetensors.torch import save
            value=save({'payload':torch.ones(4) if change=='shape' else torch.ones(3,dtype=torch.bfloat16)})
        with changed.connect() as db:
            db.execute(f'UPDATE records SET {column}=? WHERE record_id=?',(value,'0'))
    with pytest.raises((ValueError,KeyError,PermissionError)):
        TargetedPayloadStore(base,changed,frozenset({'0'})).fetch(plan)
