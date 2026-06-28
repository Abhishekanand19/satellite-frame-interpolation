const BASE = import.meta.env.VITE_API_URL || "http://localhost:8080";

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, options);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return res.json();
}

export const api = {
  status:   () => request("/api/system/status"),
  triplets: (s = 10) => request(`/api/triplets?stride=${s}&limit=200`),
  interpolate: (triplet_index, stride = 10) =>
    request("/api/interpolate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ triplet_index, stride }),
    }),
  interpolateUpload: (t0File, t2File, gtFile = null) => {
    const fd = new FormData();
    fd.append("t0_file", t0File);
    fd.append("t2_file", t2File);
    if (gtFile) fd.append("gt_file", gtFile);
    return request("/api/interpolate/upload", { method: "POST", body: fd });
  },
};