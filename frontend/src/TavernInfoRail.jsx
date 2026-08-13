import StatusPill from "./StatusPill.jsx";
import { FlameIcon, MapPinIcon, PhoneIcon } from "./icons.jsx";
import {
  TENANT_ADDRESS,
  TENANT_ATMOSPHERE,
  TENANT_HOURS,
  TENANT_PHONE,
  TENANT_PHONE_HREF,
  formatHoursLabel,
  getTodayRow,
} from "./tenant.js";

function AddressLine() {
  return (
    <p className="tavern-info__address">
      <MapPinIcon size={13} /> {TENANT_ADDRESS.line1}, {TENANT_ADDRESS.cityStateZip}
    </p>
  );
}

export default function TavernInfoRail({ variant = "rail" }) {
  const today = getTodayRow();

  if (variant === "hero") {
    return (
      <div className="tavern-info tavern-info--hero">
        <StatusPill />
        <AddressLine />
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
        <StatusPill />
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
        <AddressLine />
        <a className="tavern-info__phone" href={TENANT_PHONE_HREF}>
          <PhoneIcon size={13} /> {TENANT_PHONE}
        </a>
      </div>
    </details>
  );
}
