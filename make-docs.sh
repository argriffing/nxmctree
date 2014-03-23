
# delete parts of the old build if present
rm -rf /tmp/nxdocs
rm -rf docs/source/apidocs
rm -rf htmlcov

# copy the manually edited .rst files into the source directory
cp docs/manually-edited/*.rst docs/source/

# use sphinx to build the rest of the .rst api docs into the source directory
sphinx-apidoc --separate -o docs/source/apidocs nxmctree

# use sphinx to build a static web page from the .rst files
sphinx-build -b html docs/source /tmp/nxdocs

# remove the temporarily created rst files
rm -rf docs/source/apidocs


