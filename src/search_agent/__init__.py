def main() -> None:
    """兼容项目 console script，实际界面由 cli 包提供。"""
    from search_agent.cli.main import main as cli_main

    cli_main()
