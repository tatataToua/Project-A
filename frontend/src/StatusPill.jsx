import { ClockIcon } from "./icons.jsx";
import { formatHoursLabel, getTodayRow, isOpenNow } from "./tenant.js";

export default function StatusPill() {
  const open = isOpenNow();
  const today = getTodayRow();

  return (
    <span className={`status-pill ${open ? "status-pill--open" : "status-pill--closed"}`}>
      <ClockIcon size={13} />
      {open ? "Open now" : "Closed now"}
      {today ? ` · ${formatHoursLabel(today)}` : ""}
    </span>
  );
}
