all: build/ml_talk.pdf

texoptions = \
	     --lualatex \
	     --interaction=nonstopmode \
	     --halt-on-error \
	     --output-directory=build

build/ml_talk.pdf: FORCE | build
	latexmk $(texoptions) ml_talk.tex
	cp build/ml_talk.pdf Wendland_Sedlaczek.pdf

preview: FORCE | build
	latexmk $(texoptions) -pvc ml_talk.tex

FORCE:

build:
	mkdir -p build

clean:
	rm -r build
