install:
	"C:\Program Files\Git\bin\bash.exe" -c "./setup.sh"

sort:
	python .\scripts\classify_batch.py

pack:
	python .\scripts\postprocess_sorted.py

clean:
	python .\scripts\clean.py
