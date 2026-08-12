import { ClockIcon, FlameIcon, MapPinIcon, PhoneIcon } from "./icons.jsx";
import {
  TENANT_ADDRESS,
  TENANT_ATMOSPHERE,
  TENANT_HOURS,
  TENANT_PHONE,
  TENANT_PHONE_HREF,
  formatHoursLabel,
  getTodayRow,
  isOpenNow,
} from "./tenant.js";

export default function TavernInfoRail({ variant = "rail" }) {
  const open = isOpenNow();
  const today = getTodayRow();

  const statusPill = (
    <span className={`status-pill ${open ? "status-pill--open" : "status-pill--closed"}`}>
      <ClockIcon size={13} />
      {open ? "Open now" : "Closed now"}
      {today ? ` · ${formatHoursLabel(today)}` : ""}
    </span>
  );

  if (variant === "hero") {
    return (
      <div className="tavern-info tavern-info--hero">
        {statusPill}
        <p className="tavern-info__address">
          <MapPinIcon size={13} /> {TENANT_ADDRESS.line1}, {TENANT_ADDRESS.cityStateZip}
        </p>
      </div>
    );
  }

  return (
    <details className="tavern-rail" open>
      <summary className="tavern-rail__summary">Tavern info</summary>
      <div className="tavern-rail__body tavern-info tavern-info--rail">
        <p className="tavern-info__blurb">
          <FlameIcon size={14} /> {TENANT_ATMOSPHERE}
        </p>
        {statusPill}
        <table className="tavern-info__hours">
          <tbody>
            {TENANT_HOURS.map((row) => (
              <tr
                key={row.label}
                className={row === today ? "tavern-info__hours-row--today" : undefined}
              >
                <td>{row.label}</td>
                <td>{formatHoursLabel(row)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="tavern-info__address">
          <MapPinIcon size={13} /> {TENANT_ADDRESS.line1}, {TENANT_ADDRESS.cityStateZip}
        </p>
        <a className="tavern-info__phone" href={TENANT_PHONE_HREF}>
          <PhoneIcon size={13} /> {TENANT_PHONE}
        </a>
      </div>
    </details>
  );
}
