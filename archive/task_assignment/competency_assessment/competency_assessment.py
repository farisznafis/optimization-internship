"""
Module Name: competency_assessment.py
Objective: Implement the CompetencyAssessment class to calculate the MSG score for each employee and rank them based on their scores.

Description:
This module provides the CompetencyAssessment class which is used to assess the competencies of employees against the required competencies for various tasks. 
It calculates the weighted scores, identifies gaps between required and actual competencies, and ranks employees based on their Mean Skill Gap (MSG) scores.

Functions:
- __init__(self, rcd_df: pd.DataFrame, acd_df: pd.DataFrame) -> None: Initializes the CompetencyAssessment class with required (task) and actual (employee) competency data.
- fit(self) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]: Fits the model by calculating weights, applying them, computing gaps, applying to Qualification Space, and calculating scores.
- calculate_poc_weight(self) -> Dict[str, Dict[str, float]]: Calculates the weight or priority of each competency for each task.
- apply_weight(self, weight: Dict[str, Dict[str, float]]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, pd.Series]]: Applies the calculated weight to the required and actual competencies.
- cal_gap(self, rcd_w: Dict[str, Dict[str, float]], acd_w: Dict[str, Dict[str, pd.Series]]) -> Dict[str, Dict[str, Dict[str, float]]]: Calculates the gap between required and actual competencies.
- calculate_soq_suq(self, gap: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, List[float]]]: Calculates the sum of over-qualification (soq) and sum of under-qualification (suq) for each employee.
- calculate_MSG(self, qs: Dict[str, Dict[str, List[Any]]]) -> Dict[str, Dict[str, List[Any]]]: Calculates the Mean Skill Gap (MSG) and qualification status for each employee.
- rank_MSG(self, qs: Dict[str, Dict[str, List[Any]]]) -> Dict[str, Dict[str, float]]: Ranks the tasks for each employee based on the MSG.
- top_n_score(self, score: Dict[str, Dict[str, float]], n: float) -> Dict[str, Dict[str, float]]: Selects the top-n% tasks for each employee based on their scores.
- top_score(self, score: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]: Assigns unique estimated tasks to employees based on their highest scores.

Classes:
- CompetencyAssessment: A class to assess the competencies of employees against required competencies for tasks.

Usage:
The CompetencyAssessment class can be used to fit a model based on provided required and actual competency data, calculate the gaps, rank employees based on their competencies, and select top-n% tasks or assign tasks uniquely to employees.

Example:
>>> rcd_df = pd.DataFrame({
...     'math': [0, 3, 5, 4, 0, 4, 3, 3, 0, 5],
...     'python': [5, 3, 4, 3, 2, 1, 3, 4, 3, 5],
...     'sql': [3, 5, 4, 3, 1, 5, 4, 5, 2, 5],
...     'cloud': [4, 4, 5, 3, 0, 5, 4, 5, 0, 5],
...     'database': [4, 3, 5, 3, 1, 0, 3, 5, 2, 0],
...     'optimization': [0, 1, 5, 0, 5, 0, 4, 2, 2, 5]
... }, index=['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10'])
>>> acd_df = pd.DataFrame({
...     'math': [5, 3, 4, 4, 2],
...     'python': [5, 4, 4, 4, 3],
...     'sql': [3, 5, 4, 5, 2],
...     'cloud': [2, 4, 3, 5, 4],
...     'database': [2, 3, 4, 5, 4],
...     'optimization': [5, 5, 3, 4, 1]
... }, index=['Talent 1', 'Talent 2', 'Talent 3', 'Talent 4', 'Talent 5'])
>>> ca = CompetencyAssessment(rcd_df, acd_df)
>>> score, info = ca.fit()
>>> top_20_score = ca.top_n_score(score, 20)
>>> top_score = ca.top_score(score)

Author:
TK Bunga Matahari Team
N. Muafi, I.G.P. Wisnu N., F. Zaid N., Fauzi I.S., Joseph C.L., S. Alisya

Last Modified:
June 2024
"""

import pandas as pd
import heapq
from typing import Dict, Tuple, List, Any


class CompetencyAssessment:
    def __init__(self, rcd_df: pd.DataFrame, acd_df: pd.DataFrame) -> None:
        """
        Definition:
        RCD (Required Competency Data) - DataFrame containing Tasks Dataset.
        ACD (Acquired Competency Data) - DataFrame containing Employees Dataset.

        Initializes the CompetencyAssessment class with required (task) and actual (employee) competency data.

        Args:
            rcd_df (pd.DataFrame): DataFrame containing Tasks dataset.
            acd_df (pd.DataFrame): DataFrame containing Employees dataset.

        Example:
            >>> rcd_df = pd.DataFrame({
            ...     'math': [0, 3, 5, 4, 0, 4, 3, 3, 0, 5],
            ...     'python': [5, 3, 4, 3, 2, 1, 3, 4, 3, 5],
            ...     'sql': [3, 5, 4, 3, 1, 5, 4, 5, 2, 5],
            ...     'cloud': [4, 4, 5, 3, 0, 5, 4, 5, 0, 5],
            ...     'database': [4, 3, 5, 3, 1, 0, 3, 5, 2, 0],
            ...     'optimization': [0, 1, 5, 0, 5, 0, 4, 2, 2, 5]
            ... }, index=['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10'])
            >>> acd_df = pd.DataFrame({
            ...     'math': [5, 3, 4, 4, 2],
            ...     'python': [5, 4, 4, 4, 3],
            ...     'sql': [3, 5, 4, 5, 2],
            ...     'cloud': [2, 4, 3, 5, 4],
            ...     'database': [2, 3, 4, 5, 4],
            ...     'optimization': [5, 5, 3, 4, 1]
            ... }, index=['Talent 1', 'Talent 2', 'Talent 3', 'Talent 4', 'Talent 5'])
            >>> ca = CompetencyAssessment(rcd_df, acd_df)
        """
        self.rcd_df = rcd_df
        self.acd_df = acd_df

    def fit(self) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        """
        Fits the model by calculating weights, applying them, computing gaps, applying to Qualification Space, and calculating scores.

        Returns:
            Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
                - The computed skills metric scores of tasks for each employee.
                - A dictionary containing intermediate results (weight, weighted_rcd, weighted_acd, gap, Qualification Space, score).

        Example:
            >>> score, info = ca.fit()
        """
        weight = self.calculate_poc_weight()
        rcd_w, acd_w = self.apply_weight(weight)
        gap = self.cal_gap(rcd_w, acd_w)
        qs = self.calculate_soq_suq(gap)
        qs = self.calculate_MSG(qs)
        score = self.rank_MSG(qs)

        info = {
            "weight": weight,
            "rcd_w": rcd_w,
            "acd_w": acd_w,
            "gap": gap,
            "qs": qs,
            "score": score,
        }

        return score, info

    def calculate_poc_weight(self) -> Dict[str, Dict[str, float]]:
        """
        Calculates the weight or priority of each competency for each task.

        Returns:
            Dict[str, Dict[str, float]]: The weights or priority for each competency for each task.
        """
        required = {}
        weight = {}

        # Store each column (competency) as a separate entry in the 'required' dictionary
        for i, val in self.rcd_df.items():
            required[i] = val

        # Calculate the total required values for each task
        total_req = {}
        for competency, val in required.items():
            for task, value in val.items():
                if task not in total_req:
                    total_req[task] = 0
                total_req[task] += value

        for competency, val in required.items():
            weight[competency] = {}
            for task, value in val.items():
                if total_req[task] != 0:
                    weight[competency][task] = value / total_req[task]
                else:
                    weight[competency][task] = 0

        return weight

    def apply_weight(
        self, weight: Dict[str, Dict[str, float]]
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, pd.Series]]]:
        """
        Applies the calculated weight to the required and actual competencies.

        Args:
            weight (Dict[str, Dict[str, float]]): The weights for each competency for each task.

        Returns:
            Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, pd.Series]]]:
                - Weighted required competencies.
                - Weighted actual competencies.
        """
        weight_df = pd.DataFrame(weight)

        rcd_w = self.rcd_df.mul(weight_df, axis=1).to_dict("index")

        acd_w = {}
        for j, row_j in self.acd_df.iterrows():
            acd_w[j] = {}
            for i, row_i in weight_df.iterrows():
                acd_w[j][i] = row_j * row_i

        # Ensure that the keys are converted to strings
        rcd_w_str = {str(k): v for k, v in rcd_w.items()}
        acd_w_str = {
            str(k): {str(ki): vi for ki, vi in v.items()} for k, v in acd_w.items()
        }

        return rcd_w_str, acd_w_str

    def cal_gap(
        self, rcd_w: Dict[str, Dict[str, float]], acd_w: Dict[str, Dict[str, pd.Series]]
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Calculates the gap between required and actual competencies.

        Args:
            rcd_w (Dict[str, Dict[str, float]]): Weighted required competencies.
            acd_w (Dict[str, Dict[str, pd.Series]]): Weighted actual competencies.

        Returns:
            Dict[str, Dict[str, Dict[str, float]]]: The gap between required and actual competencies for each employee.
        """
        competency = self.rcd_df.columns.tolist()

        gap = {employee: {} for employee in self.acd_df.index}

        for employee in acd_w:
            for task in rcd_w:
                for c in competency:
                    if task not in gap[employee]:
                        gap[employee][task] = {}
                    gap[employee][task][c] = acd_w[employee][task][c] - rcd_w[task][c]

        return gap

    def calculate_soq_suq(
        self, gap: Dict[str, Dict[str, Dict[str, float]]]
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Calculates the sum of over-qualification (soq) and sum of under-qualification (suq) for each employee.

        Args:
            gap (Dict[str, Dict[str, Dict[str, float]]]): The gap between required and actual competencies for each employee.

        Returns:
            Dict[str, Dict[str, List[float]]]: The soq and suq for each employee.
        """
        qs = {}

        for employee, tasks in gap.items():
            for task, competency in tasks.items():
                soq, suq = 0, 0

                for c, value in competency.items():
                    if value >= 0:
                        soq += value
                    elif value < 0:
                        suq += value

                if employee not in qs:
                    qs[employee] = {}

                qs[employee][task] = [soq, suq]

        return qs

    def calculate_MSG(
        self, qs: Dict[str, Dict[str, List[Any]]]
    ) -> Dict[str, Dict[str, List[Any]]]:
        """
        Calculates the Mean Skill Gap (MSG) and qualification status for each employee.

        Args:
            qs (Dict[str, Dict[str, List[Any]]]): The soq and suq for each employee.

        Returns:
            Dict[str, Dict[str, List[Any]]]: The MSG and qualification status for each employee.
        """
        n = len(self.acd_df.columns)

        for employee, task_qs in qs.items():
            for task, values in task_qs.items():
                soq, suq = values[0], values[1]
                msg = (soq + suq) / n
                qs[employee][task] += [msg]

                if msg >= 0:
                    qs[employee][task] += ["Qualified"]
                else:
                    qs[employee][task] += ["Under-Qualified"]

        return qs

    def rank_MSG(
        self, qs: Dict[str, Dict[str, List[Any]]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Ranks the tasks for each employee based on the MSG.

        Args:
            qs (Dict[str, Dict[str, List[Any]]]): The MSG and qualification status for each employee.

        Returns:
            Dict[str, Dict[str, float]]: The ranked tasks for each employee.

        Example:
            >>> score, info = ca.fit()
            >>> ranked_tasks = ca.rank_MSG(info['qs'])
        """
        ranking = {}

        for employee, task_qs in qs.items():
            for task, values in task_qs.items():
                if employee not in ranking:
                    ranking[employee] = {}
                ranking[employee][task] = values[-2]

        for employee in ranking:
            ranking[employee] = dict(
                sorted(
                    ranking[employee].items(), key=lambda item: item[1], reverse=True
                )
            )

        return ranking

    def top_n_score(
        self, score: Dict[str, Dict[str, float]], n: float
    ) -> Dict[str, Dict[str, float]]:
        """
        Selects the top-n% tasks for each employee based on their scores.

        Args:
            score (Dict[str, Dict[str, float]]): The scores for each task for each employee.
            n (float): The percentage of top tasks to select.

        Returns:
            Dict[str, Dict[str, float]]: The top-n% tasks for each employee.

        Example:
            >>> top_20_score = ca.top_n_score(score, 20)
        """
        top_n = int(self.rcd_df.shape[0] * (n / 100))
        top_n_score = {}

        for employee, tasks in score.items():
            # Sort tasks by MSG score in descending order and keep only top-n%
            sorted_tasks = dict(
                sorted(tasks.items(), key=lambda item: item[1], reverse=True)[:top_n]
            )
            top_n_score[employee] = sorted_tasks

        return top_n_score

    def top_score(
        self, score: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Assigns unique estimated tasks to employees based on their highest scores.

        Args:
            score (Dict[str, Dict[str, float]]): The scores for each task for each employee.

        Returns:
            Dict[str, Dict[str, float]]: The unique assigned tasks for each employee

        Example:
            >>> top_score = ca.top_score(score)
        """
        assigned_tasks = {}
        task_assignment = {talent: {} for talent in score}
        all_tasks = {task for tasks in score.values() for task in tasks}
        remaining_tasks = all_tasks.copy()
        talent_heap = []

        # Initialize heap with talents and their max scores
        for talent in score:
            heapq.heappush(talent_heap, (-max(score[talent].values()), talent))

        # Assign tasks based on highest scores first
        while talent_heap and remaining_tasks:
            _, talent = heapq.heappop(talent_heap)
            for task, task_score in sorted(
                score[talent].items(), key=lambda item: item[1], reverse=True
            ):
                if task in remaining_tasks and task_score >= 0:
                    task_assignment[talent][task] = task_score
                    assigned_tasks[task] = (talent, task_score)
                    remaining_tasks.remove(task)
                    break

        # Ensure every talent has at least one task
        for talent in score:
            if not task_assignment[talent]:
                for task, task_score in sorted(
                    score[talent].items(), key=lambda item: item[1], reverse=True
                ):
                    if task not in assigned_tasks:
                        task_assignment[talent][task] = task_score
                        assigned_tasks[task] = (talent, task_score)
                        remaining_tasks.remove(task)
                        break

        # Assign remaining tasks (including those with scores < 0)
        while remaining_tasks:
            for talent in score:
                if not remaining_tasks:
                    break
                for task, task_score in sorted(
                    score[talent].items(), key=lambda item: item[1], reverse=True
                ):
                    if task in remaining_tasks:
                        task_assignment[talent][task] = task_score
                        assigned_tasks[task] = (talent, task_score)
                        remaining_tasks.remove(task)
                        break

        return task_assignment
