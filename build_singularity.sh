docker build -f Dockerfile.artifact -t local/spreg:latest .
rm spreg.sif
singularity build spreg.sif docker-daemon://local/spreg:latest
