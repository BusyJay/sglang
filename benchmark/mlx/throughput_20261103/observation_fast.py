"""Preplanned C-level getters retain complete graph invalidation checks."""
from operator import attrgetter, itemgetter

def getter(keys, attribute=False):
    fn=(attrgetter if attribute else itemgetter)(*keys)
    if len(keys)==1:
        return lambda value:(fn(value),)
    return fn

def prepare(plan):
    checks=[]
    for value,keys,length,observed in plan.nodes:
        if keys is None:
            scalar=[(k,p) for k,s,p in observed if s]
            identities=[(k,p) for k,s,p in observed if not s]
            checks.append((value,keys,length,scalar,identities,None,None))
            continue
        mapping=dict(value) if isinstance(value,dict) else {}
        mapping.update(vars(value) if hasattr(value,'__dict__') else {})
        defaults=[(k,s,p) for k,s,p in observed if k not in mapping]
        scalars={k:mapping[k] for k,s,p in observed if s and k in mapping}
        identities={k:p for k,s,p in observed if not s and k in mapping}
        checks.append((value,keys,length,scalars,identities,mapping,defaults))
    plan._getter_checks=checks

def current(plan,modules):
    if modules is not plan.modules:return False
    for cls,fields in plan.classes.items():
        if vars(cls)!=fields:return False
    for value,keys,length,scalars,identities,mapping,defaults in plan._getter_checks:
        if keys is None:
            if len(value)!=length:return False
            for key,previous in scalars:
                child=value[key]
                if type(child) is not type(previous) or child!=previous:return False
            for key,previous in identities:
                if id(value[key])!=previous:return False
            continue
        if keys!=(tuple(value) if isinstance(value,dict) else (),tuple(vars(value)) if hasattr(value,'__dict__') else ()):
            return False
        actual=dict(value) if isinstance(value,dict) else {}
        actual.update(vars(value) if hasattr(value,'__dict__') else {})
        for key,previous in identities.items():
            if id(actual[key])!=previous:return False
            actual[key]=mapping[key]
        # Restore excluded values; they are outside graph equations.
        for key in mapping.keys()-scalars.keys()-identities.keys():actual[key]=mapping[key]
        if actual!=mapping:return False
        for key,previous in scalars.items():
            if type(actual[key]) is not type(previous):return False
        for key,scalar,previous in defaults:
            child=getattr(value,key)
            if scalar:
                if type(child) is not type(previous) or child!=previous:return False
            elif id(child)!=previous:return False
    return True
