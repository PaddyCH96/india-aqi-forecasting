# Weather Coordinates

How each of the six cities is mapped to an ERA5 reanalysis grid point, how far
apart those two things are, and where the substitution is weakest.

## The Mapping

The coordinate table lives in `lib.config.CITY_COORDS` — one dict, six cities,
each with a `lat` and a `lon`. The values were promoted verbatim from the table
`scripts/fetch_recent_aqi.py` had been using since the Open-Meteo air-quality
pull, so ERA5 weather and any future CAMS air-quality pull describe the same
point rather than drifting apart. There is exactly one coordinate table in this
repository; anything that needs a city's location imports it.

The archive is queried with `models=era5` pinned explicitly (the Open-Meteo
default resolves to a seamless ERA5/ERA5-Land blend at a *different* grid point)
and `timezone=Asia/Kolkata`, so every daily aggregate covers an IST calendar day.
The grid latitude, longitude and elevation the API actually returned are
persisted on every `city_weather` row as `grid_latitude`, `grid_longitude` and
`grid_elevation_m` — the mapping below is therefore auditable from the data
itself, not only from this document.

## City to Grid Point

Measured live against the archive API on 2026-09-01.

| City | Requested lat/lon | ERA5 grid point | Grid elevation | Distance |
|------|-------------------|-----------------|----------------|----------|
| Delhi | 28.6139 / 77.2090 | 28.50 / 77.25 | 214 m | 13.3 km |
| Mumbai | 19.0760 / 72.8777 | 19.00 / 73.00 | 6 m | 15.4 km |
| Bengaluru | 12.9716 / 77.5946 | 13.00 / 77.50 | 910 m | 10.7 km |
| Chennai | 13.0827 / 80.2707 | 13.00 / 80.25 | 12 m | 9.5 km |
| Hyderabad | 17.3850 / 78.4867 | 17.50 / 78.50 | 505 m | 12.9 km |
| Kolkata | 22.5726 / 88.3639 | 22.50 / 88.25 | 12 m | 14.2 km |

## Resolution Context

ERA5 is a 0.25° × 0.25° grid — roughly 28 km north–south by 26 km east–west at
Indian latitudes. A point can therefore sit at most about 19 km from the centre
of the cell containing it, and all six cities land comfortably inside a single
cell. The distances above are what snapping to the nearest cell centre costs;
none of them is anomalous.

## Limitations

The grid point is not the city. It is worth being blunt about what that means
before any of these features are read as meteorology.

CPCB's city-level AQI is an aggregate over multiple monitoring stations spread
across an urban area tens of kilometres wide. ERA5 supplies a single
cell-average value that is neither the city centre nor any particular station.
The two series are being joined because they describe roughly the same region on
the same day, not because they describe the same place.

Three specific caveats:

- **Mumbai.** The grid point sits roughly 15 km east of the island city, on the
  mainland side of **Thane Creek**. Sea-breeze onset, humidity and the coastal
  temperature moderation recorded there are not the island's. Of the six, this
  is the mapping most likely to mislead.
- **Bengaluru (910 m) and Hyderabad (505 m).** Grid elevation dominates surface
  pressure, so `surface_pressure_hpa` for these cities is near-constant and
  offset from the coastal cities by a fixed amount. In a pooled model it will
  behave as a city identifier rather than as a meteorological signal, and should
  be treated with that suspicion.
- **Urban effects.** No city's urban heat island, street-canyon ventilation or
  local emission-driven microclimate is resolved by ERA5 at this resolution, in
  any of the six cells. What is captured is the regional synoptic condition the
  city sits inside.

## Why This Is Not Fixed Here

This representativeness loss is inherent to using a global reanalysis as a
city-level covariate; there is no coordinate tweak that removes it. Whether it
actually costs anything is an empirical question, and the honest way to answer
it is Phase 3's skill measurement — if weather features do not improve forecast
skill over the AQI-only baseline, this document is the first place to look.
