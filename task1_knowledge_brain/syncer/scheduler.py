"""Small stdlib daily scheduler used by the Task 1 API process."""
from datetime import datetime, time, timedelta
import logging
import os
import threading
from zoneinfo import ZoneInfo

from .daily_sync import DailySitemapSyncer

logger = logging.getLogger(__name__)


class DailySyncScheduler:
    def __init__(self, syncer=None, run_time=None, timezone_name=None):
        self.syncer = syncer or DailySitemapSyncer()
        self.run_time = run_time or os.getenv("KB_DAILY_SYNC_TIME", "03:00")
        self.timezone_name = timezone_name or os.getenv("KB_DAILY_SYNC_TIMEZONE", "Asia/Kolkata")
        self.stop_event = threading.Event()
        self.thread = None

    def next_run(self, now=None):
        zone = ZoneInfo(self.timezone_name)
        now = (now or datetime.now(zone)).astimezone(zone)
        hour, minute = (int(x) for x in self.run_time.split(":", 1))
        due = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return due if due > now else due + timedelta(days=1)

    def _run(self):
        while not self.stop_event.is_set():
            due = self.next_run()
            if self.stop_event.wait(max(1, (due - datetime.now(due.tzinfo)).total_seconds())):
                return
            try:
                report = self.syncer.sync()
                logger.info("Scheduled sitemap sync status=%s message=%s", report.status, report.message)
            except Exception:
                logger.exception("Scheduled sitemap sync failed")

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._run, name="task1-daily-sync", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
