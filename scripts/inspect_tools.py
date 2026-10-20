"""Show public Tool schemas alongside Runtime-only effect metadata."""

from resolv.tools.registry import TOOL_REGISTRY


def main() -> None:
    for name, registration in TOOL_REGISTRY.items():
        print(name)
        print(f"  effect: {registration.effect.value}")
        print(f"  description: {registration.tool.description.splitlines()[0]}")
        print("  args:")
        for argument, schema in registration.tool.args.items():
            print(f"    {argument}: {schema.get('type', 'unknown')}")


if __name__ == "__main__":
    main()
