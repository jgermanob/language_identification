#!/bin/sh

mkdir "corpora"
cd "corpora" || exit
mkdir "openSubtitles"
cd "openSubtitles" || exit
mkdir "raw_files"
cd "raw_files" || exit

wget --content-disposition "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/en.txt.gz"
wget --content-disposition "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/es.txt.gz"
wget --content-disposition "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/fr.txt.gz"
wget --content-disposition "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/de.txt.gz"
wget --content-disposition "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/it.txt.gz"
wget --content-disposition "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/pt.txt.gz"
wget --content-disposition "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/tr.txt.gz"
wget --content-disposition "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/nl.txt.gz"
wget --content-disposition "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/sv.txt.gz"

gunzip ./*

