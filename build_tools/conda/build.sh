# Install the r-packages
./install_r_packages.sh
# Install tigramite
$PYTHON setup.py install --single-version-externally-managed --record=record.txt
echo $DEBUG
ls $PREFIX
