<h1>Disclaimer</h1>
I made this tool for fun to try and see if it can solve minesweeper online intermediates.

<h2>What it can and cannot do</h2>
It can auto set flags where the bombs are. <br/>
It can click where the safe spots are. <br/>
It cannot solve edge cases because I found it wasn't worth my time. <br />

<h2>Setup</h2>
I highly recommend venv <br />
<b>I expect `https://minesweeper.online/` and to be playing intermediate and zoom size of 28. This is important. </b>


1. `pip install -r -requirements.txt`
1. Make sure you select a few to give it the program a heads start. I will not randomly select spots for you.
1. `python3 solver.py`

Video showing how I gave it a start first, then I ran the script on another screen.
It can solve the guaranteeds, but on some edge cases, it will stop. The user will have to manually set themselves. They can start the program again once it finds cases it can solve.

https://youtu.be/nybezTAA8yo