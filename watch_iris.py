
import time
import subprocess
import sys
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

PROJECT_DIR = Path(__file__).resolve().parent
CSV_FILE = (PROJECT_DIR / "iris.csv").resolve()

class IrisCSVHandler(FileSystemEventHandler):
    def __init__(self):
        self.last_run = 0

    def on_modified(self, event):
        if event.is_directory:
            return

        if Path(event.src_path).resolve() == CSV_FILE:
            # Ignore repeated file-save events occurring close together.
            if time.time() - self.last_run < 3:
                return

            self.last_run = time.time()
            print("\n iris.csv changed! Running DVC pipeline...")

            result = subprocess.run(
                [sys.executable, "-m", "dvc", "repro"],
                cwd=PROJECT_DIR
            )

            if result.returncode == 0:
                print("Pipeline completed successfully!")
            else:
                print("Pipeline failed. Check the error messages above.")

if __name__ == "__main__":
    event_handler = IrisCSVHandler()
    observer = Observer()
    observer.schedule(event_handler, str(PROJECT_DIR), recursive=False)
    observer.start()

    print("Watching iris.csv for changes...")
    print("Keep this terminal open. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
