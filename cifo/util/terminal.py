# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------------------------
"""
Print Fancy
-------------------
Content

 ▶ def print_box( messages, size = BOX_SIZE, font_color = None)

 ▶ def prepare_line( message, size = BOX_SIZE, font_color = None )
 
 ▶ def print_line( message = "", size = BOX_SIZE )
 
 ▶ class FontColor

─────────────────────────────────────────────────────────────────────────

CIFO - Computation Intelligence for Optimization

Author: Fernando A J Peres - fperes@novaims.unl.pt - (2019) version L4.0

"""
# -------------------------------------------------------------------------------------------------

import locale

BOX_SIZE = 100


class Terminal:
    @staticmethod
    def clear():
        print(chr(27) + "[2j")
        print("\033c")
        print("\x1bc")

    # Print Box
    # --------------------------------------------------------------------------------------------------
    @staticmethod
    def print_box(messages, size=BOX_SIZE, font_color=None):
        line_element = "─"
        line = ""

        # line size
        for _ in range(0, size - 4):
            line += line_element

        if font_color:
            print(font_color, end="")

        # open
        print(f"┌─{line}─┐")

        # content
        for message in messages:
            empty_spaces = Terminal.prepare_line(message)
            print(f"│  {message}{empty_spaces}│")

        # close
        print(f"└─{line}─┘\033[0m")

    # Print Line
    # --------------------------------------------------------------------------------------------------
    @staticmethod
    def prepare_line(message, size=BOX_SIZE, font_color=None):
        line_element = "─"
        line = ""

        for _ in range(0, size - 4):
            line += line_element

        empty_spaces = ""
        for _ in range(0, size - 4 - len(message)):
            empty_spaces += " "

        return empty_spaces

    # Print Line
    # --------------------------------------------------------------------------------------------------
    @staticmethod
    def print_line(message="", size=BOX_SIZE):
        if message != "":
            print(f"   { message }")

        line_element = "─"
        line = " "
        for _ in range(0, size - 2):
            line += line_element
        print(line)

    # --------------------------------------------------------------------------------------------------
    # FontColor


# --------------------------------------------------------------------------------------------------
class FontColor:
    Red = "\033[0;31;48m"
    Green = "\033[0;32;48m"
    Yellow = "\033[0;33;48m"
    Blue = "\033[0;34;48m"
    Purple = "\033[0;35;48m"
    Cyan = "\033[0;36;48m"
    White = "\033[0;37;48m"  # 40 - black | 48m - dark gray


#
# The above ANSI escape code will set the text colour to bright green. The format is;
# \033[  Escape code, this is always the same
# 1 = Style, 1 for normal.
# 32 = Text colour, 32 for bright green.
# 40m = Background colour, 40 is for black.
# TEXT COLOR	CODE	TEXT STYLE	CODE	BACKGROUND COLOR	CODE
# Black	        30	    No effect	0	    Black	            40
# Red	        31	    Bold	    1	    Red	                41
# Green	        32	    Underline	2	    Green	            42
# Yellow	    33	    Negative1	3	    Yellow	            43
# Blue	        34	    Negative2	5	    Blue	            44
# Purple	    35			                Purple	            45
# Cyan	        36			                Cyan	            46
# White	        37			                White	            47
#
#
# RED   = "\033[1;31m"
# BLUE  = "\033[1;34m"
# CYAN  = "\033[1;36m"
# GREEN = "\033[0;32m"
# RESET = "\033[0;0m"
# BOLD    = "\033[;1m"
# REVERSE = "\033[;7m"
#
