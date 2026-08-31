import ast, pathlib
for p in sorted(pathlib.Path("src").rglob("*.py")):
    try:
        tree = ast.parse(p.read_text(encoding="utf-8"))
    except SyntaxError:
        continue
    names = [f"  {'class' if isinstance(n, ast.ClassDef) else 'def'} "
             f"{n.name}({', '.join(a.arg for a in n.args.args)})"
             if not isinstance(n, ast.ClassDef) else f"  class {n.name}"
             for n in tree.body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    if names:
        print(f"\n{p}")
        print("\n".join(names))