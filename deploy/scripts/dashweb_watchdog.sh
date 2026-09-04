#!/usr/bin/env bash
# dashweb-watchdog.sh — restart dashweb.service when its /_healthz probe stops
# responding. Catches HANGS (process alive but unresponsive) that systemd's
# Restart=always cannot, because a hung process never exits.
set -u

URL="http://127.0.0.1:8050/_healthz"
POLL_S=15
CURL_TIMEOUT_S=6
THRESHOLD=3                 # consecutive failures before restarting
MIN_RESTART_GAP_S=90        # never restart more often than this

failcount=0
last_restart=0

while true; do
  now=$(date +%s)
  if curl -fs -o /dev/null -m "$CURL_TIMEOUT_S" "$URL"; then
    if [ "$failcount" -gt 0 ]; then
      logger -t dashweb-watchdog "healthz recovered ($failcount consecutive failures cleared)"
      failcount=0
    fi
  else
    failcount=$((failcount + 1))
    logger -t dashweb-watchdog "healthz check failed ($failcount/$THRESHOLD)"
    if [ "$failcount" -ge "$THRESHOLD" ]; then
      if [ $((now - last_restart)) -ge "$MIN_RESTART_GAP_S" ]; then
        logger -t dashweb-watchdog "restarting dashweb.service"
        systemctl restart dashweb.service
        last_restart=$now
        failcount=0
      else
        logger -t dashweb-watchdog "restart skipped (too soon after last restart)"
      fi
    fi
  fi
  sleep "$POLL_S"
done