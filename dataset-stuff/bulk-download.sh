
wget --recursive \
    --no-parent \
    --no-host-directories \
    --reject "index.html*" \
    --continue \
    --execute robots=off \
    --wait=1 \
    --random-wait \
    https://www.sidc.be/uset/data/cameras/output_img/FTS/L1c/USET_White_Light/
