# Oncall playbook (sample)

## Severity levels

SEV1: customer-facing outage or data loss risk — page immediately, open a war room, update status page every 30 minutes.

SEV2: major degradation with workaround — notify stakeholders within 15 minutes, aim for mitigation within 4 hours.

## Handoff

End-of-shift summary must include open incidents, risky deploys, and noisy alerts to tune. Use the handoff template in the wiki.

## Rollbacks

If error rates double after a deploy within 10 minutes, initiate rollback unless the incident commander decides otherwise. Document timeline in the incident ticket.
