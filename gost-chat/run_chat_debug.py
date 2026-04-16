from launcher_core import LauncherConfig, run_launcher


if __name__ == "__main__":
    raise SystemExit(run_launcher(LauncherConfig(debug=True, reload=True)))
