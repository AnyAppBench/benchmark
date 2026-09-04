"""Temporary ntodotxt task shims for app generalization.

These classes expose app-specific task names while reusing canonical
Tasks.org information-retrieval evaluators. This removes unsupported skips for
ntodotxt while a full app-native task-state port is developed.
"""

from android_world.task_evals.information_retrieval import information_retrieval
from android_world.task_evals.information_retrieval import information_retrieval_registry


_IR_TASKS = information_retrieval_registry.InformationRetrievalRegistry[
    information_retrieval.InformationRetrieval
]().registry


class TasksCompletedTasksForDateForNtodotxt(_IR_TASKS["TasksCompletedTasksForDate"]):
  """Temporary shim for ntodotxt using canonical TasksCompletedTasksForDate."""


class TasksDueNextWeekForNtodotxt(_IR_TASKS["TasksDueNextWeek"]):
  """Temporary shim for ntodotxt using canonical TasksDueNextWeek."""


class TasksDueOnDateForNtodotxt(_IR_TASKS["TasksDueOnDate"]):
  """Temporary shim for ntodotxt using canonical TasksDueOnDate."""


class TasksHighPriorityTasksForNtodotxt(_IR_TASKS["TasksHighPriorityTasks"]):
  """Temporary shim for ntodotxt using canonical TasksHighPriorityTasks."""


class TasksHighPriorityTasksDueOnDateForNtodotxt(
    _IR_TASKS["TasksHighPriorityTasksDueOnDate"]
):
  """Temporary shim for ntodotxt using canonical TasksHighPriorityTasksDueOnDate."""


class TasksIncompleteTasksOnDateForNtodotxt(_IR_TASKS["TasksIncompleteTasksOnDate"]):
  """Temporary shim for ntodotxt using canonical TasksIncompleteTasksOnDate."""

