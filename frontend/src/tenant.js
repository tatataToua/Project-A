export const TENANT_SLUG = "two-owls-tavern";
export const TENANT_NAME = "Two Owls Tavern";
export const TENANT_LOCATION = "Millbrook, Oregon";

export const TENANT_TAGLINE =
  "Wood-fired, farm-close, unpretentious — New American tavern fare in a " +
  "century-old Millbrook hardware store.";

export const TENANT_ATMOSPHERE =
  "Exposed brick, the original tin ceiling, and a wood-fired oven built into " +
  'the old loading dock — named for the great horned owls that nested in the ' +
  'eaves during renovation and, per co-founder Mara Whitfield, "supervised ' +
  'the whole thing."';

export const TENANT_ADDRESS = {
  line1: "214 Elm Street",
  cityStateZip: "Millbrook, OR 97xxx",
};

export const TENANT_PHONE = "(555) 019-2288";
export const TENANT_PHONE_HREF = "tel:+15550192288";

// days[] uses Date.getDay() convention: 0=Sun ... 6=Sat
export const TENANT_HOURS = [
  { days: [1], label: "Monday", open: null, close: null },
  { days: [2, 3, 4], label: "Tue–Thu", open: "11:30", close: "21:30" },
  { days: [5, 6], label: "Fri–Sat", open: "11:30", close: "22:00" },
  { days: [0], label: "Sunday", open: "11:30", close: "21:00" },
];

export const TENANT_EXAMPLE_QUESTIONS = [
  "Do I need a reservation?",
  "Where do I park?",
  "Do you have a private room?",
  "Is the patio dog-friendly?",
  "Do you have vegan or gluten-free options?",
  "Can I order takeout?",
];

export function getTodayRow(date = new Date()) {
  const day = date.getDay();
  return TENANT_HOURS.find((row) => row.days.includes(day));
}

// Simplification: uses posted open/close times, not the "kitchen closes 30
// min early" nuance from the tenant's about.md.
export function isOpenNow(date = new Date()) {
  const row = getTodayRow(date);
  if (!row || !row.open) return false;
  const minutesNow = date.getHours() * 60 + date.getMinutes();
  const [oh, om] = row.open.split(":").map(Number);
  const [ch, cm] = row.close.split(":").map(Number);
  return minutesNow >= oh * 60 + om && minutesNow < ch * 60 + cm;
}

export function formatHoursLabel(row) {
  if (!row.open) return "Closed";
  const fmt = (t) => {
    const [h, m] = t.split(":").map(Number);
    const period = h >= 12 ? "PM" : "AM";
    const h12 = h % 12 === 0 ? 12 : h % 12;
    return m === 0 ? `${h12} ${period}` : `${h12}:${String(m).padStart(2, "0")} ${period}`;
  };
  return `${fmt(row.open)} – ${fmt(row.close)}`;
}
