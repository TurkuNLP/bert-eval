curl -o fi_tdt-ud-train.conllu https://raw.githubusercontent.com/UniversalDependencies/UD_Finnish-TDT/master/fi_tdt-ud-train.conllu

cat fi_tdt-ud-train.conllu | grep -P '^# text =' | perl -pe 's/# text = //g' > finnish_sentences.txt
