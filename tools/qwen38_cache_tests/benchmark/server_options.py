"""Read CLI type/arity from the checked-out ServerArgs namespace declarations.

No torch/model imports. Custom parsers/actions use the actual ServerArgs CLI when
available; fail closed if an offline host cannot import those dependencies.
"""
from __future__ import annotations
import argparse
import ast
from functools import lru_cache
from .core import ROOT


@lru_cache(maxsize=1)
def schema():
    declarations = {}
    for path in sorted((ROOT / 'python/sglang/srt/arg_groups/fields').glob('*.py')):
        tree = ast.parse(path.read_text())
        for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
            for node in cls.body:
                if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
                    continue
                name = node.target.id
                annotation = node.annotation
                if isinstance(annotation, ast.Subscript) and ast.unparse(annotation.value) == 'A':
                    parts = annotation.slice.elts if isinstance(annotation.slice, ast.Tuple) else [annotation.slice]
                    typ, metadata = parts[0], parts[1]
                else:
                    typ, metadata = annotation, None
                kwargs = {k.arg:k.value for k in metadata.keywords} if isinstance(metadata, ast.Call) else {}
                if 'no_cli' in kwargs and ast.literal_eval(kwargs['no_cli']):
                    continue
                flags = ['--' + name.replace('_','-')]
                if 'cli_name' in kwargs:
                    flags[0] = ast.literal_eval(kwargs['cli_name'])
                if 'aliases' in kwargs:
                    flags += ast.literal_eval(kwargs['aliases'])
                if isinstance(typ, ast.Subscript) and ast.unparse(typ.value) == 'Optional':
                    typ = typ.slice
                origin = ast.unparse(typ.value) if isinstance(typ, ast.Subscript) else ast.unparse(typ)
                nargs, choices = None, None
                if origin in ('List','list'):
                    typ = typ.slice; nargs = '+'
                elif origin == 'Literal':
                    values = typ.slice.elts if isinstance(typ.slice, ast.Tuple) else [typ.slice]
                    choices = [ast.literal_eval(v) for v in values]
                    typ = ast.Name(id=type(choices[0]).__name__)
                type_name = ast.unparse(typ)
                supported = type_name in ('bool','str','int','float') and not any(k in kwargs for k in ('type_parser','action','action_kwargs'))
                boolean_optional = 'action' in kwargs and ast.unparse(kwargs['action']) == 'argparse.BooleanOptionalAction'
                if boolean_optional:
                    supported = True
                    flags += ['--no-' + flags[0][2:]]
                if 'nargs' in kwargs:
                    nargs = ast.literal_eval(kwargs['nargs'])
                custom = ast.unparse(kwargs['type_parser']) if 'type_parser' in kwargs else None
                if custom in ('human_readable_int','nullable_str','json_list_type','json.loads'):
                    supported = True
                    nargs = None
                if type_name == 'bool' and custom is None:
                    nargs = 0
                spec = {'dest':name, 'type':type_name, 'nargs':nargs, 'choices':choices, 'supported':supported, 'canonical':flags[0], 'boolean_optional':boolean_optional, 'custom':custom}
                for flag in flags:
                    declarations[flag] = spec
    # Convenience graph CLI is registered explicitly, outside field metadata.
    for phase in ('prefill','decode'):
        declarations['--cuda-graph-backend-'+phase] = {'dest':'cuda_graph_backend_'+phase,'type':'str','nargs':None,'choices':['disabled','full','breakable','tc_piecewise'],'supported':True,'canonical':'--cuda-graph-backend-'+phase}
    return declarations


@lru_cache(maxsize=1)
def actual_parser():
    from sglang.srt.server_args import ServerArgs
    parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    ServerArgs.add_cli_args(parser)
    for action in parser._actions:
        action.required = False
    return parser


@lru_cache(maxsize=1)
def custom_parsers():
    # Compile the exact pure parsing functions from local production source.
    # No importing utils.common (torch/model dependencies) on offline CPU hosts.
    import json
    import re
    from decimal import Decimal
    from types import SimpleNamespace
    tree = ast.parse((ROOT/'python/sglang/srt/utils/common.py').read_text())
    functions = [n for n in tree.body if isinstance(n,ast.FunctionDef)
                 and n.name in ('human_readable_int','nullable_str','json_list_type')]
    scope = {'argparse':argparse,'re':re,'Decimal':Decimal,'json':json,
             'orjson':SimpleNamespace(loads=json.loads)}
    exec(compile(ast.Module(body=functions,type_ignores=[]),'production CLI parsers','exec'),scope)
    return {name:scope[name] for name in ('human_readable_int','nullable_str','json_list_type')} | {'json.loads':json.loads}


def parse_options(argv):
    """Return canonical flag -> (complete argv group, actual typed assertion)."""
    result, i = {}, 0
    declarations = schema()
    while i < len(argv):
        arg = argv[i]
        if not isinstance(arg, str) or not arg.startswith('--'):
            raise ValueError('server option group requires a known --flag')
        key, equals, value = arg.partition('=')
        if key in ('--model-path','--model','--revision','--host','--port','--model-loader-extra-config'):
            raise ValueError('managed server argument override')
        spec = declarations.get(key)
        if spec is None:
            raise ValueError('unknown/unregistered server option: '+key)
        i += 1
        values = [value] if equals else []
        # Consume the whole group before merging, including multiple URLs.
        while i < len(argv) and not argv[i].startswith('--'):
            values.append(argv[i]); i += 1
        nargs = spec['nargs']
        valid = ((nargs == 0 and not values) or
                 (nargs is None and len(values) == 1) or
                 (nargs == '+' and len(values) >= 1) or nargs == '*' or
                 (nargs == '?' and len(values) <= 1) or
                 (type(nargs) is int and len(values) == nargs))
        if not valid:
            raise ValueError('invalid server option arity: '+key)
        canonical = spec['canonical']
        group = [key if spec.get('boolean_optional') else canonical, *values]
        if spec['supported']:
            convert = custom_parsers()[spec['custom']] if spec.get('custom') else {'str':str,'int':int,'float':float,'bool':bool}[spec['type']]
            try:
                typed = [convert(v) for v in values]
            except (ValueError, argparse.ArgumentTypeError) as exc:
                raise ValueError('invalid server option value: '+key) from exc
            if spec['choices'] is not None and any(v not in spec['choices'] for v in typed):
                raise ValueError('invalid server option choice: '+key)
            expected = (not key.startswith('--no-')) if nargs == 0 else typed if nargs in ('+','*') or type(nargs) is int else typed[0] if typed else None
        else:
            try:
                parser = actual_parser()
                parsed = parser.parse_args(group)
                expected = getattr(parsed,spec['dest'])
            except (ImportError, SystemExit, argparse.ArgumentError, AttributeError) as exc:
                raise ValueError('server option needs actual CLI parser/dependencies: '+key) from exc
        result[canonical] = (group, {spec['dest']:expected})
    return result
