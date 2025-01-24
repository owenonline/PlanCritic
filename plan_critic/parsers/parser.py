"""
This file contains the abstract definition of a class to handle parsing output
from planners.  Primarily, this serves to define the interface required by the
planner component to handle parsing output form planners.
"""

import abc
from typing import Optional, Union


class Parser(abc.ABC):
	"""
	A parser is defined as a class that injests strings describing a plan, and
	generates a dictionary representation of plans when available.

	Methods
	-------

	Attributes
	----------
	"""


	def __init__(self):
		"""
		"""

		pass


	@abc.abstractmethod
	def injest(self, line: str) -> Optional[dict]:
		"""
		Injest a line to parse.  If a plan can be produced at this point, then
		a dictionary representation of the completed plan should be provided.
		Otherwise, `None` should be returned
		"""

		raise NotImplementedError


	@abc.abstractmethod
	def complete(self) -> Optional[dict]:
		"""
		Called when the underlying planner has completed running.  Used to 
		provide any final plans yet to be processed.
		"""

		raise NotImplementedError
