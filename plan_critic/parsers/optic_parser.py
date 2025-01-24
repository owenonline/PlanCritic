"""
This file contains the definition of a parser to handle output from the optic 
planner.
"""

import re
from typing import Optional, Union

from .parser import Parser


class OpticParser(Parser):
	"""
	OpticParser is a parser designed to process command-line output from the
	optic planner.


	Methods
	-------

	Attributes
	----------
	"""

	# optic will prepend solution information with a semicolon (;), followed by
	# a set of plan steps.  Of interest to this parser are lines starting with
	# a semicolon which contain interesting plan information, as well as plan
	# steps.  Interesting plan information takes these following forms:
	#     ; Plan found with metric 0.025
	#     ; Cost: 0.025
	#     ; States evaluated so far: 32
	#     ; States evaluated: 505163
	#     ; Time 175.75
	#     ;;;; Solution Found
	# optic plan steps look like:
	#     0.011: (move rover1 waypoint5 waypoint2)  [0.001]
	# The regular expression to match this should try to match the decimal
	# value, colon, an arbitrary sequence of characters, then another decimal
	# between two square brackets.
	# NOTE:  I'm assuming that the two decimal values will have at least one
	#        leading zero, and one trailing zero.  This may not be the case, 
	#        in which chase this expression would need to be modified to have
	#        0 or more decimals as appropriate.
	__PLAN_STEP_PATTERN = r"(?P<time_step>((\d+)\.(\d+))): *(?P<action>\(.*\)) *\[(?P<duration>((\d+)\.(\d+)))\]"
	__STATES_EVALUATED_PATTERN = r"; States evaluated.*: *(?P<states_evaluated>\d+)"
	__COST_PATTERN = r"; (Plan found with metric|Cost:) *(?P<cost>((\d+).(\d+)))"
	__TIME_PATTERN = r"; Time *(?P<time>((\d+).(\d+)))"
	__FINAL_SOLUTION_PATTERN = r";+ Solution Found *"


	def __init__(self):
		"""
		"""

		# Initialize a buffer to store lines as they come in, and a buffer for
		# processing current plans (which may not be fully complete when )
		self._buffer = []

		# Create regular expression patterns from the strings
		self.__plan_step_pattern = re.compile(self.__PLAN_STEP_PATTERN)
		self.__states_evaluated_pattern = re.compile(self.__STATES_EVALUATED_PATTERN)
		self.__cost_pattern = re.compile(self.__COST_PATTERN)
		self.__time_pattern = re.compile(self.__TIME_PATTERN)
		self.__final_solution_pattern = re.compile(self.__FINAL_SOLUTION_PATTERN)


	def process_buffer(self) -> Optional[dict]:
		"""
		Processes a plan buffer to convert the contents to a plan dicitonary
		"""

		# Generate a list of matches for each pattern that can appear in the
		# planer output, removing any matches that are empty
		actions = [ self.__plan_step_pattern.match(line) 
		            for line in self._buffer ]
		actions = [ action for action in actions if action is not None ]

		states_evaluated = [ self.__states_evaluated_pattern.match(line)
		                     for line in self._buffer ]
		states_evaluated = [ match for match in states_evaluated 
		                     if match is not None ]

		costs = [ self.__cost_pattern.match(line) for line in self._buffer ]
		costs = [ cost for cost in costs if cost is not None ]

		times = [ self.__time_pattern.match(line) for line in self._buffer ]
		times = [ time for time in times if time is not None ]

		final_solutions = [ self.__final_solution_pattern.match(line) 
		                    for line in self._buffer ]
		final_solutions = [ solution for solution in final_solutions 
		                    if solution is not None ]

		# Now that the buffer has been injested, empty it out so that it can
		# be ready for the next plan
		self._buffer.clear()

		# If there are no actions, then there isn't a plan in the buffer, so
		# simply return None
		if len(actions) == 0:
			return None

		# Construct the plan dictionary with the contents of the matches
		plan = { }
		plan['actions'] = [ action.groupdict() for action in actions ]

		if len(states_evaluated) == 0:
			# TODO:  issue a warning.
			plan['states_evaluated'] = -1
		else:
			plan['states_evaluated'] = states_evaluated[0].groupdict()['states_evaluated']

		if len(costs) == 0:
			# TODO:  issue a warning
			plan['cost'] = -1
		else:
			plan['cost'] = costs[0].groupdict()['cost']

		if len(times) == 0:
			# TODO:  issue a warning
			plan['compute_time'] = -1
		else:
			plan['compute_time'] = times[0].groupdict()['time']

		# This solution is final if any of the lines matched the regex
		plan['is_final'] = (len(final_solutions) > 0)


		return plan


	def injest(self, line: str) -> Optional[dict]:
		"""
		Add a line to the buffer, checking to see if a plan is ready to parse.
		If the buffer contains a plan ready to parse, then return a dictionary
		representation of the plan, otherwise, return `None`
		"""

		# Depending on the contents of the provided line and the contents of
		# the buffer, one of three things may happen:
		#
		# 1.  The line is recognized as associated with a plan, and should be
		#     added to the buffer;
		# 2.  The line is not associated with a plan, and the buffer is empty,
		#     so it should be discarded; and
		# 3.  The line is not associated with a plan, and the buffer is not 
		#     empty, so the line should be discarded and the buffer processed.


		# 1.  Should the line be added to the current plan buffer?  For 
		#     optic, plans can be easily parsed if a) they start with a
		#     semicolon, or b) if they're of the form 
		#     {float}: {action} [{float}], and the current plan buffer is
		#     non-empty
		if line.startswith(';'):
			self._buffer.append(line)
			return None

		elif self.__plan_step_pattern.match(line) is not None:
			self._buffer.append(line)
			return None

		# 2.  If the buffer is currently empty, then there's not plan to 
		#     process.  Simply discard this line
		elif len(self._buffer) == 0:
			return None

		# 3.  Otherwise a) the buffer is not empty, and b) this line does
		#     not match a plan line, so the planning is done!
		else:
			return self.process_buffer()

	def injest_stdout(self, stdout: str) -> Optional[dict]:
		"""
		Add a line to the buffer, checking to see if a plan is ready to parse.
		If the buffer contains a plan ready to parse, then return a dictionary
		representation of the plan, otherwise, return `None`
		"""

		# Split the stdout into lines, and process each line
		lines = stdout.split('\n')
		lines.append("fin") # add something that the parser isnt looking for to trigger processing
		for line in lines:
			result = self.injest(line)
			if result is not None:
				return result

		# If we've gotten to this point, then there's no plan to parse
		return None

	def complete(self) -> Optional[dict]:
		"""
		Called when the underlying planner has completed running.  Used to 
		provide any final plans yet to be processed
		"""

		return self.process_buffer()
