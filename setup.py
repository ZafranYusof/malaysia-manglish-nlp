"""Setup for manglish-nlp package."""

from setuptools import setup, find_packages

setup(
    name='manglish-nlp',
    version='2.0.0',
    author='Zafran',
    author_email='zafran@example.com',
    description='Natural Language Processing toolkit for Malaysian Manglish',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ZafranYusof/manglish-nlp',
    packages=find_packages(),
    package_data={
        'manglish_nlp': ['resources/*.json'],
    },
    entry_points={
        'console_scripts': [
            'manglish=manglish_nlp.__main__:main',
        ],
    },
    python_requires='>=3.8',
    install_requires=[],  # Zero dependencies for core
    extras_require={
        'transformers': ['torch>=2.0', 'transformers>=4.30', 'sentencepiece'],
        'embeddings': ['gensim>=4.0'],
        'spacy': ['spacy>=3.0'],
        'all': ['torch>=2.0', 'transformers>=4.30', 'sentencepiece', 'gensim>=4.0', 'spacy>=3.0'],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Natural Language :: Malay',
    ],
    keywords='nlp malay manglish malaysian text-processing sentiment tokenizer stemmer pos-tagger ner dialect',
)
