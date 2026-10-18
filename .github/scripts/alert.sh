#!/usr/bin/env bash
# Opens a GitHub issue when a scheduled job fails, so the failure reaches Gary by email, and
# closes it after the next clean run. A repeat failure adds a comment instead of a new issue.
#
#   alert.sh open "<issue title>" [file with details]
#   alert.sh close "<issue title>"
#
# Needs GH_TOKEN (the job's GITHUB_TOKEN with issues: write).
set -euo pipefail
action=$1 title=$2 details=${3:-}
run="$GITHUB_SERVER_URL/$GITHUB_REPOSITORY/actions/runs/$GITHUB_RUN_ID"
number=$(gh issue list --repo "$GITHUB_REPOSITORY" --state open --search "in:title \"$title\"" \
  --json number,title --jq "map(select(.title == \"$title\"))[0].number // empty")

if [ "$action" = open ]; then
  body="This run needs a look: $run"
  if [ -n "$details" ] && [ -s "$details" ]; then
    body+=$'\n\n'"$(cat "$details")"
  fi
  if [ -n "$number" ]; then
    gh issue comment "$number" --repo "$GITHUB_REPOSITORY" --body "Failed again. $body"
  else
    gh issue create --repo "$GITHUB_REPOSITORY" --title "$title" \
      --body "$body"$'\n\n'"This issue closes itself after the next clean run. cc @Garygunn94"
  fi
elif [ -n "$number" ]; then
  gh issue close "$number" --repo "$GITHUB_REPOSITORY" --comment "The next run went through cleanly: $run"
fi
