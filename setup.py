"""Setup for malaysian-manglish-nlp package."""

from setuptools import setup, find_packages, Extension

try:
    from setuptools.command.build_ext import build_ext as _build_ext

    class BuildExtOptional(_build_ext):
        """Allow C extension build to fail gracefully."""

        def run(self):
            try:
                super().run()
            except Exception:
                print("WARNING: C extension failed to build, falling back to pure Python.")

        def build_extension(self, ext):
            try:
                super().build_extension(ext)
            except Exception:
                print(f"WARNING: Failed to build {ext.name}, using Python fallback.")

except ImportError:
    BuildExtOptional = None

ext_modules = [
    Extension(
        'malaysian_manglish_nlp._tokenizer_fast',
        sources=['malaysian_manglish_nlp/_tokenizer_fast.c'],
    )
]

setup(
    name='malaysian-manglish-nlp',
    version='3.1.0',
    author='Zafran',
    author_email='zafran@example.com',
    description='Natural Language Processing toolkit for Malaysian Manglish',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ZafranYusof/malaysia-manglish-nlp',
    packages=find_packages(),
    package_data={
        'malaysian_manglish_nlp': ['resources/*.json'],
    },
    entry_points={
        'console_scripts': [
            'manglish=malaysian_manglish_nlp.__main__:main',
        ],
    },
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtOptional} if BuildExtOptional else {},
    python_requires='>=3.8',
    install_requires=[],
    extras_require={
        'transformers': ['torch>=2.0', 'transformers>=4.30', 'sentencepiece'],
        'embeddings': ['gensim>=4.0'],
        'spacy': ['spacy>=3.0'],
        'all': ['torch>=2.0', 'transformers>=4.30', 'sentencepiece', 'gensim>=4.0', 'spacy>=3.0'],
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
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
