"""
This module defines classes for the plan critic component within the ONR CAI agent.
The module itself contains functionality converse with a user and update a plan
based on the user's feedback.
"""

# Assume the package is installed, and set the version number based on the
# installed package version number.  If it cannot, then default the version
# number to 0.0.0
try:
	from importlib.metadata import version
	__VERSION__ = version("marine-cadastre-environment")
except Exception as e:
	__VERSION__ = "0.0.0"