
# delete parts of the old build if present
rm -rf /tmp/nxdocs
rm -f docs/source/*.rst

# copy the manually edited .rst files into the source directory
cp docs/manually-edited/*.rst docs/source/

# use sphinx to build the rest of the .rst api docs into the source directory
sphinx-apidoc --separate -o docs/source nxmctree

# use sphinx to build a static web page from the .rst files
sphinx-build -b html docs/source /tmp/nxdocs

