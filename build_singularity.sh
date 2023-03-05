docker build -f Dockerfile.artifact -t local/spreg:latest .
sudo singularity build spreg.sif docker-daemon://local/spreg:latest
