from launcher_core import LauncherConfig, USER_LOG_FILE, run_launcher


if __name__ == "__main__":
    raise SystemExit(run_launcher(LauncherConfig(debug=False, log_file=USER_LOG_FILE)))
