#!/usr/bin/env bash
# ws-monitor-watchdog.sh — restart the dashboard service when its /_healthz
# probe stops responding. Catches HANGS (process alive but unresponsive) that
# systemd's Restart=always cannot, because a hung process never exits.
#
# Restarts ws-monitor-dashboard.service by default. DASHWEB_SERVICE remains an
# override for deployments that predate the unit renaming.
set -u

URL="http://127.0.0.1:8050/_healthz"
SERVICE="${DASHWEB_SERVICE:-ws-monitor-dashboard.service}"
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
      logger -t ws-monitor-watchdog "healthz recovered ($failcount consecutive failures cleared)"
      failcount=0
    fi
  else
    failcount=$((failcount + 1))
    logger -t ws-monitor-watchdog "healthz check failed ($failcount/$THRESHOLD)"
    if [ "$failcount" -ge "$THRESHOLD" ]; then
      if [ $((now - last_restart)) -ge "$MIN_RESTART_GAP_S" ]; then
        if systemctl restart "$SERVICE"; then
          logger -t ws-monitor-watchdog "restarted $SERVICE"
          last_restart=$now
          failcount=0
        else
          # Restart failed: keep the failure counter so we retry next poll
          # instead of latching into a silent 90s gap.
          logger -t ws-monitor-watchdog "restart of $SERVICE FAILED (exit $?)"
        fi
      else
        logger -t ws-monitor-watchdog "restart skipped (too soon after last restart)"
      fi
    fi
  fi
  sleep "$POLL_S"
done