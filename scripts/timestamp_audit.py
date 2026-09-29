from datetime import datetime, timezone

utc_now = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
local_now = datetime.now().isoformat()
tz_offset = str(datetime.now().astimezone().utcoffset())
gen_method = 'datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")'

print('utc_now:', utc_now)
print('local_now:', local_now)
print('timezone_offset:', tz_offset)
print('generation_method:', gen_method)
print('ends_with_Z:', utc_now.endswith('Z'))