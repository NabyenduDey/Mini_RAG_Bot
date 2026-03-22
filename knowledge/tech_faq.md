# Engineering FAQ (sample)

## VPN

Install the corporate VPN client from the internal software catalog. If authentication fails, reset the MFA token in the identity portal and retry. Split tunneling is disabled for security.

## Git workflow

Use trunk-based development with short-lived feature branches. All merges require one review and passing CI. Force-push to `main` is blocked.

## Databases

Staging credentials are rotated weekly and posted in the secrets manager. Never commit connection strings. Use read-only users for analytics queries.
