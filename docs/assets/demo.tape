# vhs tape for the ANEForge README hero animation.
# Records examples/demo.py running live on the ANE, then convert the GIF to an APNG
# (which autoplays and loops reliably). Regenerate from the repo root with:
#   vhs docs/assets/demo.tape
#   ffmpeg -y -i docs/assets/demo.gif -plays 0 docs/assets/demo.png
# Requires: vhs, ffmpeg, a built dispatch dylib (sh aneforge/_lib/build.sh), an Apple
# Silicon Mac with an ANE. python3.11 is the interpreter that has numpy + aneforge.
# demo.gif is the intermediate (gitignored); demo.png (APNG) is the shipped asset.

Output docs/assets/demo.gif

Set Shell bash
Set FontSize 19
Set Width 1280
Set Height 700
Set Padding 28
Set Margin 0
Set TypingSpeed 55ms
Set Framerate 24
Set Theme { "background": "#0d1117", "foreground": "#c9d1d9", "cursor": "#eb8246", "selection": "#264f78", "black": "#484f58", "red": "#ff7b72", "green": "#3fb950", "yellow": "#d29922", "blue": "#58a6ff", "magenta": "#bc8cff", "cyan": "#39c5cf", "white": "#b1bac4", "brightBlack": "#6e7681", "brightRed": "#ffa198", "brightGreen": "#56d364", "brightYellow": "#e3b341", "brightBlue": "#79c0ff", "brightMagenta": "#d2a8ff", "brightCyan": "#56d4dd", "brightWhite": "#f0f6fc" }

# Type the command while hidden, then reveal, so frame 0 already shows
# "$ python examples/demo.py" (meaningful even if a browser freezes the GIF
# on its first frame under prefers-reduced-motion).
Hide
Type "export PROMPT_COMMAND=; PS1='\033[38;2;110;118;126m$\033[0m '; alias python=python3.11"
Enter
Type "clear"
Enter
Type "python examples/demo.py"
Show
Sleep 1400ms
Enter
Sleep 10500ms
Sleep 1800ms
