//! `strftime_now`'s formatting: C's `strftime`, for the conversions chat templates use,
//! in UTC.
//!
//! # Why UTC, and why written here
//!
//! `transformers` calls Python's `datetime.now().strftime(format)`, which is local time.
//! Local time needs the machine's time zone database, and std does not read it; a crate
//! that does would be a dependency for one line of one template. A prompt is also built
//! at compile time here, on a build machine whose zone says nothing about where the
//! program runs, so local time would not be more right, only different. UTC is the
//! same everywhere, and [`Chat::with_time`](crate::Chat::with_time) pins it outright.
//!
//! The conversions are the C locale's. Llama 3.1 and later ask for `%d %b %Y`; the rest
//! are here because they are cheap and a template could plausibly use them. An unknown
//! conversion is printed as written, which is what glibc does.

/// Format `secs`, seconds since the Unix epoch, as UTC.
pub(crate) fn strftime(format: &str, secs: i64) -> String {
    let days = secs.div_euclid(86_400);
    let sod = secs.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let (hour, minute, second) = (sod / 3600, sod % 3600 / 60, sod % 60);
    // 1970-01-01 was a Thursday; 0 is Sunday, as `%w` counts.
    let weekday = (days + 4).rem_euclid(7) as usize;
    let yday = day_of_year(year, month, day);

    let mut out = String::new();
    let mut chars = format.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        // glibc's `-` flag drops the zero padding: `%-d` is `5`, not `05`.
        let pad = if chars.peek() == Some(&'-') {
            chars.next();
            false
        } else {
            true
        };
        let num = |n: i64, width: usize| if pad { format!("{n:0width$}") } else { n.to_string() };
        let Some(conv) = chars.next() else {
            out.push('%');
            break;
        };
        let text = match conv {
            'a' => WEEKDAYS[weekday][..3].to_owned(),
            'A' => WEEKDAYS[weekday].to_owned(),
            'b' | 'h' => MONTHS[month as usize - 1][..3].to_owned(),
            'B' => MONTHS[month as usize - 1].to_owned(),
            'd' => num(day.into(), 2),
            'e' => if pad { format!("{day:2}") } else { day.to_string() },
            'm' => num(month.into(), 2),
            'y' => num(year.rem_euclid(100), 2),
            'Y' => year.to_string(),
            'j' => num(yday, 3),
            'H' => num(hour, 2),
            'I' => num(if hour % 12 == 0 { 12 } else { hour % 12 }, 2),
            'M' => num(minute, 2),
            'S' => num(second, 2),
            'p' => if hour < 12 { "AM" } else { "PM" }.to_owned(),
            'w' => weekday.to_string(),
            'u' => if weekday == 0 { 7 } else { weekday }.to_string(),
            'F' => format!("{year}-{month:02}-{day:02}"),
            'D' => format!("{month:02}/{day:02}/{:02}", year.rem_euclid(100)),
            'T' => format!("{hour:02}:{minute:02}:{second:02}"),
            'R' => format!("{hour:02}:{minute:02}"),
            'c' => format!(
                "{} {} {day:2} {hour:02}:{minute:02}:{second:02} {year}",
                &WEEKDAYS[weekday][..3],
                &MONTHS[month as usize - 1][..3]
            ),
            'x' => format!("{month:02}/{day:02}/{:02}", year.rem_euclid(100)),
            'X' => format!("{hour:02}:{minute:02}:{second:02}"),
            'Z' => "UTC".to_owned(),
            'z' => "+0000".to_owned(),
            's' => secs.to_string(),
            'n' => "\n".to_owned(),
            't' => "\t".to_owned(),
            '%' => "%".to_owned(),
            other => {
                if pad {
                    format!("%{other}")
                } else {
                    format!("%-{other}")
                }
            }
        };
        out.push_str(&text);
    }
    out
}

const WEEKDAYS: [&str; 7] = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
const MONTHS: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];

/// Days since 1970-01-01 to a proleptic Gregorian `(year, month, day)`. Howard
/// Hinnant's `civil_from_days`, which is exact over the whole `i64` day range that
/// matters here, with no table and no loop.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = yoe + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

fn day_of_year(year: i64, month: u32, day: u32) -> i64 {
    const BEFORE: [i64; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    BEFORE[month as usize - 1] + i64::from(day) + i64::from(leap && month > 2)
}
