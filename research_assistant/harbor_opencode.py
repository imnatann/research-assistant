"""Install the cached OpenCode binary during Harbor's agent setup phase."""

from pathlib import Path

from harbor.agents.installed.opencode import OpenCode


class CachedOpenCode(OpenCode):
    async def install(self, environment):
        await self.ensure_system_dependencies(environment, ("coreutils",))
        machine = await self.exec_as_agent(environment, command="uname -m")
        arch = {"aarch64": "arm64", "x86_64": "x64"}.get((machine.stdout or "").strip())
        if arch is None:
            raise ValueError(f"Unsupported Linux OpenCode architecture: {machine.stdout!r}")
        cache = Path(__file__).resolve().parents[1] / ".runtime" / "agentic-bench" / "opencode"
        binary = cache / f"{arch}-{self.options.version}"
        await environment.upload_file(binary, "/tmp/harbor-opencode")
        await self.exec_as_root(
            environment, command="install -m 0755 /tmp/harbor-opencode /usr/local/bin/opencode",
        )
        if not await self.is_requested_version_installed(environment):
            raise ValueError(f"OpenCode {self.options.version} is required in the trial")
