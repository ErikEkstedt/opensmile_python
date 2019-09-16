#!/bin/bash

# Downloads opensmile into ./opensmile/opensmile-2.3.0 
# and adds custom config files for GeMap features 50/10 ms

# Create opensmile directory
mkdir -p opensmile 

url=https://www.audeering.com
link=https://www.audeering.com/download/opensmile-2-3-0-tar-gz/?wpdmdl=4782
target=opensmile/opensmile-2.3.0.tar.gz 

echo "Downloading opensmile-2.3.0.tar.gz"
wget -O $target --referer $url $link --no-check-certificate


echo "Extracting Opensmile and removing tar"
tar -zxf $target -C opensmile/
rm $target

echo
echo "Example:"
echo "SMILExtract -C $CONF50 -I /path/to/wav -D /path/to/output"
echo
