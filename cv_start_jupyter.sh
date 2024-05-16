# GOTTA START AN INTERACT SESSION
module load cuda
ipnip=$(hostname -i)
ipnport=8889
echo "Paste the following command onto your local computer:"
echo "ssh -N -L ${ipnport}:${ipnip}:${ipnport} $USER@sshcampus.ccv.brown.edu"
output = $(python -m notebook --no-browser --port=$ipnport --ip=$ipnip)
