# Acknowledgement

malaysian-manglish-nlp stands on the shoulders of many excellent open-source projects and the broader Malaysian NLP community.

---

## Tools & Libraries

| Project | Role |
|---------|------|
| [spaCy](https://spacy.io) | NLP pipeline architecture inspiration |
| [Malaya](https://github.com/huseinzol05/Malaya) | Pioneer Malaysian NLP toolkit  -  primary design reference |
| [HuggingFace Transformers](https://huggingface.co) | XLM-Roberta model and fine-tuning framework |
| [HuggingFace Datasets](https://huggingface.co/datasets) | Dataset loading and management |
| [Gensim](https://radimrehurek.com/gensim/) | Word2Vec and FastText training |
| [NetworkX](https://networkx.org) | Graph algorithms for text graph analysis |
| [NumPy](https://numpy.org) | Numerical computation backbone |
| [pandas](https://pandas.pydata.org) | Dataset manipulation |
| [FastAPI](https://fastapi.tiangolo.com) | REST API server |
| [pytest](https://pytest.org) | Testing framework |

---

## Data Sources

| Source | Usage |
|--------|-------|
| Twitter/X Malaysian users | Sentiment training data |
| Lowyat forum posts | Sentiment and slang data |
| Reddit r/malaysia | Sentiment data |
| Malaysian news portals | NER annotation source |
| [DBP (Dewan Bahasa dan Pustaka)](https://prpm.dbp.gov.my) | Dictionary and standard Malay reference |
| [Bernama](https://www.bernama.com) | News text for NER |
| [Berita Harian](https://www.bharian.com.my) | News text for NER |

---

## Research & Models

| Resource | Contribution |
|----------|-------------|
| Mesolitica NanoT5 | Baseline Malay sentiment model for comparison |
| [huseinzol05/malaysian-sentiment](https://github.com/huseinzol05/Malaya/tree/master/malaya/sentiment) | Sentiment analysis benchmark |
| XLM-Roberta base | Base model for fine-tuned classifier (v3.2.0) |
| DistilBERT multilingual | Previous base model (v3.0.0-v3.1.0) |
| CoNLL-2003 NER format | Annotation standard for NER dataset |

---

## Academic Support

- **UMP (Universiti Malaysia Pahang)**  -  Academic institution and Final Year Project supervision
- Faculty of Computing, UMP  -  Infrastructure and guidance

---

## Community

- Malaysian NLP community  -  feedback, dataset contributions, and bug reports
- Open source contributors on GitHub  -  pull requests, issues, and suggestions
- r/malaysia and Lowyat community members whose public posts (anonymised) form part of the training data

---

## Inspiration

malaysian-manglish-nlp was heavily inspired by [Malaya](https://github.com/huseinzol05/Malaya) by Hussein Zolkepli. Malaya pioneered the Malaysian NLP toolkit space and set the standard for API design, documentation quality, and model coverage. This project aims to complement Malaya by focusing specifically on the Manglish code-switching register.

---

## References

1. Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L., & Stoyanov, V. (2020). Unsupervised cross-lingual representation learning at scale. *Proceedings of ACL 2020*, 8440–8451.
2. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. *arXiv preprint arXiv:1910.01108*.
2. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. *arXiv preprint arXiv:1301.3781*.
3. Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). Enriching word vectors with subword information. *Transactions of the ACL*, 5, 135–146.
4. Tjong Kim Sang, E. F., & De Meulder, F. (2003). Introduction to the CoNLL-2003 shared task: Language-independent named entity recognition. *Proceedings of CoNLL-2003*, 142–147.
