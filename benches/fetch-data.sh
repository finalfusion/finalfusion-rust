#!/bin/sh

cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -f de-structgram-20190426-opq.fifu ]; then
  curl -O https://blob.danieldk.eu/sticker-models/de-structgram-20190426-opq.fifu
fi

if [ ! -f dewiki-aa-00.txt ]; then
  curl http://www.sfs.uni-tuebingen.de/a3-public-data/tueba-ddp/r4/dewiki-201901/parts/AA/wiki_00.zst | unzstd - | cut -f 2 > dewiki-aa-00.txt
fi
