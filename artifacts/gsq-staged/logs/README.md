# Durable execution logs

Each .log.gz is a lossless archive of the corresponding raw .log cited in the run records. Decompress it to recover the exact bytes and verify the recorded raw-log SHA256. Raw .log files remain locally available but are not tracked.
