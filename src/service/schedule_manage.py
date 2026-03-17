from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Iterable, List, Tuple

from data.postgres import get_connection


DEFAULT_SLOT_MINUTES = 30
SLOT_START_TIMES: List[time] = [
	time(8, 30),
	time(9, 0),
	time(9, 30),
	time(10, 0),
	time(10, 30),
	time(11, 0),
	time(11, 30),
	time(13, 30),
	time(14, 0),
	time(14, 30),
	time(15, 0),
	time(15, 30),
	time(16, 0),
]


def _available_slots(day: datetime.date, booked: Iterable[Tuple[datetime, datetime]], duration_minutes: int) -> List[datetime]:
	"""Return allowed slot starts on the day that are not booked."""

	block = timedelta(minutes=duration_minutes)
	allowed = [datetime.combine(day, slot) for slot in SLOT_START_TIMES]
	free: List[datetime] = []

	for slot_start in allowed:
		slot_end = slot_start + block
		conflict = any(not (slot_end <= start or slot_start >= end) for start, end in booked)
		if not conflict:
			free.append(slot_start)

	return free


def _lookup_doctor_id(cur, doctor_name: str) -> Tuple[int | None, str | None]:
	"""Resolve doctor_id from doctor name; return (id, error_message)."""

	cur.execute(
		"""
		SELECT id
		FROM doctors
		WHERE name ILIKE %s
		ORDER BY id
		""",
		(doctor_name,),
	)
	rows = cur.fetchall()
	if not rows:
		return None, f"Doctor '{doctor_name}' not found."
	if len(rows) > 1:
		return None, f"Multiple doctors match '{doctor_name}'. Please specify."
	return rows[0][0], None


def _to_naive(dt: datetime) -> datetime:
	"""Strip timezone info so comparisons remain offset-naive."""

	return dt.replace(tzinfo=None) if dt.tzinfo else dt


def _normalize_requested_start(requested_start: datetime | str) -> Tuple[datetime | None, str | None]:
	"""Accept datetime or 'dd-mm-YYYY HH:MM' string; return naive datetime or error."""

	if isinstance(requested_start, str):
		try:
			parsed = datetime.strptime(requested_start, "%d-%m-%Y %H:%M")
		except ValueError:
			return None, "Requested start time format must be 'dd-mm-YYYY HH:MM'."
		return parsed, None

	if isinstance(requested_start, datetime):
		return _to_naive(requested_start), None

	return None, "Requested start time must be a datetime or a string."


def book_doctor_appointment(
	doctor_name: str,
	patient_name: str,
	requested_start: datetime | str,
	duration_minutes: int = DEFAULT_SLOT_MINUTES,
) -> str:
	"""
	Book an appointment if the slot is free; otherwise return alternatives.

	Expected schema (adjust if your table differs):
		doctors(id serial primary key, name text unique)
		appointments(id serial primary key,
				 doctor_id int not null references doctors(id),
				 patient_name text,
				 appointment_start timestamp,
				 appointment_end timestamp,
				 status text default 'booked')
	"""
	requested_started, normalize_err = _normalize_requested_start(requested_start)
	if normalize_err:
		return normalize_err
	requested_started = _to_naive(requested_started)
	conn = get_connection()
	if not conn:
		return "Unable to connect to the schedule database."

	day = requested_started.date()
	block = timedelta(minutes=duration_minutes)
	requested_end = requested_started + block

	allowed_starts = {datetime.combine(day, slot) for slot in SLOT_START_TIMES}
	if requested_started not in allowed_starts:
		allowed_str = ", ".join(t.strftime("%H:%M") for t in SLOT_START_TIMES)
		return f"Requested time must match available slots: {allowed_str}."

	try:
		with conn:
			with conn.cursor() as cur:
				doctor_id, lookup_err = _lookup_doctor_id(cur, doctor_name)
				if lookup_err:
					return lookup_err

				cur.execute(
					"""
					SELECT appointment_start, appointment_end
					FROM appointments
					WHERE doctor_id = %s
					  AND appointment_start::date = %s
					ORDER BY appointment_start
					""",
					(doctor_id, day),
				)
				booked = [(_to_naive(row[0]), _to_naive(row[1])) for row in cur.fetchall()]

				conflict = any(
					not (requested_end <= start or requested_started >= end)
					for start, end in booked
				)

				if not conflict:
					cur.execute(
						"""
						INSERT INTO appointments (doctor_id, patient_name, appointment_start, appointment_end, status)
						VALUES (%s, %s, %s, %s, 'booked')
						RETURNING id
						""",
						(doctor_id, patient_name, requested_started, requested_end),
					)
					appt_id = cur.fetchone()[0]
					return (
						f"Appointment confirmed with doctor {doctor_id} "
						f"on {requested_started.strftime('%Y-%m-%d %H:%M')} (ref #{appt_id})."
					)

				free_slots = _available_slots(day, booked, duration_minutes)
				if free_slots:
					slot_text = ", ".join(slot.strftime("%H:%M") for slot in free_slots)
					return (
						f"Requested time is unavailable. Available times on {day}: "
						f"{slot_text}."
					)

				return (
					f"{day} is fully booked for doctor {doctor_id}. "
					"Please choose another date or doctor."
				)

	except Exception as exc:
		return f"Error while booking appointment: {exc}"
	finally:
		conn.close()
