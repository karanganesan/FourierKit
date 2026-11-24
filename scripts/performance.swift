// builds docs/performance.html from docs/performance.jsonl, golden/manifest.json and docs/tries.jsonl.
// run it from anywhere:  swift scripts/performance.swift
// foundation only. older log lines may lack fields, so every read can come back empty.

import Foundation

typealias Object = [String: Any]

let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
let clipNames = ["short", "medium", "long"]

// MARK: reading

func readLines(_ url: URL) -> [Object] {
    guard let text = try? String(contentsOf: url, encoding: .utf8) else { return [] }
    var out: [Object] = []
    for line in text.split(whereSeparator: \.isNewline) {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty, let data = trimmed.data(using: .utf8),
              let object = (try? JSONSerialization.jsonObject(with: data)) as? Object else { continue }
        out.append(object)
    }
    return out
}

func readObject(_ url: URL) -> Object {
    guard let data = try? Data(contentsOf: url),
          let object = (try? JSONSerialization.jsonObject(with: data)) as? Object else { return [:] }
    return object
}

func value(_ object: Object?, _ keys: String...) -> Any? {
    var current: Any? = object
    for key in keys { current = (current as? Object)?[key] }
    return current
}

func number(_ object: Object?, _ keys: String...) -> Double? {
    var current: Any? = object
    for key in keys { current = (current as? Object)?[key] }
    return (current as? NSNumber)?.doubleValue
}

func text(_ object: Object?, _ keys: String...) -> String? {
    var current: Any? = object
    for key in keys { current = (current as? Object)?[key] }
    if let string = current as? String { return string }
    if let n = current as? NSNumber { return n.stringValue }
    return nil
}

func passed(_ run: Object) -> Bool { (run["passed"] as? Bool) ?? false }

// MARK: formatting

func escape(_ string: String) -> String {
    string.replacingOccurrences(of: "&", with: "&amp;")
        .replacingOccurrences(of: "<", with: "&lt;")
        .replacingOccurrences(of: ">", with: "&gt;")
        .replacingOccurrences(of: "\"", with: "&quot;")
}

let none = "n/a"

func fixed(_ v: Double?, _ digits: Int = 2) -> String {
    guard let v, v.isFinite else { return none }
    return String(format: "%.\(digits)f", v)
}

func seconds(_ v: Double?) -> String {
    guard let v, v.isFinite else { return none }
    if v < 0.01 { return String(format: "%.2f ms", v * 1000) }
    if v < 1 { return String(format: "%.0f ms", v * 1000) }
    return String(format: "%.2f s", v)
}

func times(_ v: Double?) -> String {
    guard let v, v.isFinite else { return none }
    return String(format: "%.2fx", v)
}

func percent(_ v: Double?) -> String {
    guard let v, v.isFinite else { return none }
    return String(format: "%.0f%%", v * 100)
}

func gigabytes(_ v: Double?) -> String {
    guard let v, v.isFinite else { return none }
    return String(format: "%.2f gb", v / 1e9)
}

func whole(_ v: Double?) -> String {
    guard let v, v.isFinite else { return none }
    return String(format: "%.0f", v)
}

func date(_ run: Object) -> String {
    guard let raw = text(run, "date") else { return none }
    let parts = raw.split(separator: "T")
    guard parts.count == 2 else { return raw.lowercased() }
    let time = parts[1].replacingOccurrences(of: "Z", with: "").prefix(5)
    return "\(parts[0]) \(time) utc"
}

func label(_ run: Object) -> String {
    (text(run, "label") ?? "no label").lowercased()
}

func thermal(_ v: Double?) -> String {
    guard let v else { return none }
    switch Int(v) {
    case 0: return "nominal"
    case 1: return "fair"
    case 2: return "serious"
    case 3: return "critical"
    default: return "state \(Int(v))"
    }
}

func placement(_ run: Object) -> String {
    guard let map = run["placement"] as? [String: Any], !map.isEmpty else { return none }
    let order = ["mel", "encoder", "decoder", "search"]
    let known = order.filter { map[$0] != nil }
    let rest = map.keys.filter { !order.contains($0) }.sorted()
    return (known + rest).map { "\($0) \(String(describing: map[$0] ?? ""))" }.joined(separator: ", ")
}

func mark(_ ok: Bool) -> String {
    ok ? "<span class=\"mark pass\">passed</span>" : "<span class=\"mark fail\">failed</span>"
}

// MARK: tables

func row(_ title: String, _ cells: [String]) -> String {
    let head = "<th scope=\"row\">\(title)</th>"
    return "<tr>\(head)" + cells.map { "<td class=\"num\">\($0)</td>" }.joined() + "</tr>"
}

func table(_ label: String, head: [String], rows: [String]) -> String {
    let heads = head.enumerated().map { index, name in
        "<th scope=\"col\"\(index == 0 ? "" : " class=\"num\"")>\(name)</th>"
    }.joined()
    return """
    <div class="table-wrap" tabindex="0" role="region" aria-label="\(escape(label))"><table>
    <thead><tr>\(heads)</tr></thead>
    <tbody>
    \(rows.joined(separator: "\n"))
    </tbody>
    </table></div>
    """
}

// MARK: chart

struct Series {
    var name: String
    var cssClass: String
    var values: [Double?]
}

func chart(series: [Series], passes: [Bool], title: String) -> String {
    let count = passes.count
    let width = 960.0, height = 320.0
    let left = 48.0, right = 24.0, top = 16.0, bottom = 40.0
    let all = series.flatMap { $0.values.compactMap { $0 } }.filter { $0.isFinite }
    guard count > 0, !all.isEmpty else { return "<p class=\"dim\">no values to plot yet.</p>" }

    // a round top for the axis
    let peak = max(all.max() ?? 1, 0.1)
    let step: Double = peak <= 1 ? 0.2 : peak <= 2 ? 0.5 : peak <= 5 ? 1 : peak <= 10 ? 2 : peak <= 20 ? 5 : 10
    let ceiling = (peak / step).rounded(.up) * step
    let plotWidth = width - left - right, plotHeight = height - top - bottom

    func x(_ index: Int) -> Double {
        count == 1 ? left + plotWidth / 2 : left + plotWidth * Double(index) / Double(count - 1)
    }
    func y(_ v: Double) -> Double { top + plotHeight * (1 - v / ceiling) }

    var svg = "<svg viewBox=\"0 0 \(Int(width)) \(Int(height))\" role=\"img\" aria-label=\"\(escape(title))\"><title>\(escape(title))</title>"
    var tick = 0.0
    while tick <= ceiling + step / 1000 {
        let line = y(tick)
        svg += "<line class=\"grid\" x1=\"\(left)\" x2=\"\(width - right)\" y1=\"\(fixed(line, 1))\" y2=\"\(fixed(line, 1))\"/>"
        svg += "<text x=\"\(left - 8)\" y=\"\(fixed(line + 4, 1))\" text-anchor=\"end\">\(fixed(tick, step < 1 ? 1 : 0))x</text>"
        tick += step
    }
    // one label per run, thinned so they never overlap
    let every = max(1, Int((Double(count) / 24).rounded(.up)))
    for index in 0..<count where index % every == 0 || index == count - 1 {
        svg += "<text x=\"\(fixed(x(index), 1))\" y=\"\(height - 12)\" text-anchor=\"middle\">\(index + 1)</text>"
    }
    for (number, one) in series.enumerated() {
        let tone = "p\(number + 1)"
        // a gap in the values breaks the line
        var path = ""
        var drawing = false
        for (index, v) in one.values.enumerated() {
            guard let v, v.isFinite else { drawing = false; continue }
            path += "\(drawing ? "L" : "M")\(fixed(x(index), 1)) \(fixed(y(v), 1)) "
            drawing = true
        }
        if !path.isEmpty { svg += "<path class=\"\(one.cssClass)\" d=\"\(path)\" stroke-linejoin=\"round\"/>" }
        for (index, v) in one.values.enumerated() {
            guard let v, v.isFinite else { continue }
            let ok = index < passes.count ? passes[index] : false
            let state = ok ? "full" : "hollow"
            svg += "<circle class=\"\(state) \(tone)\" cx=\"\(fixed(x(index), 1))\" cy=\"\(fixed(y(v), 1))\" r=\"4\"><title>run \(index + 1), \(escape(one.name)): \(fixed(v))x\(ok ? "" : ", failed")</title></circle>"
        }
    }
    svg += "</svg>"
    return "<div class=\"chart\">\(svg)</div>"
}

// MARK: page

/// live replay numbers from the newest modes line, and where the climb stands from docs/standing.jsonl
func extraSections(live: Object?, standing: [Object]) -> String {
    var body = ""
    if let live, let clips = live["live"] as? [String: Object] {
        body += "<section aria-labelledby=\"live\"><h2 id=\"live\">live replay</h2>\n"
        body += "<p class=\"dim\">each golden clip played in at real speed through <code>listen</code>, from the run labelled \(escape(label(live))). settled text was never taken back, and every final phrase equals <code>transcribe</code> on the same split points.</p>\n"
        var rows: [String] = []
        for name in ["short", "medium", "long"] {
            guard let clip = clips[name] else { continue }
            rows.append(row(name, [
                seconds(number(clip, "first_text")), seconds(number(clip, "median_delay")), seconds(number(clip, "worst_delay")),
                whole(number(clip, "updates")), whole(number(clip, "draft_changes")), whole(number(clip, "phrases")), seconds(number(clip, "finish_wait")),
            ]))
        }
        body += table("live replay by clip", head: ["clip", "first text", "median draft delay", "worst delay", "updates", "draft changes", "phrases", "finish waited"], rows: rows)
        body += "</section>\n"
    }
    if !standing.isEmpty {
        body += "<section aria-labelledby=\"standing\"><h2 id=\"standing\">where it stands</h2>\n"
        var rows: [String] = []
        for entry in standing {
            rows.append("<tr><th scope=\"row\">\(escape(text(entry, "topic") ?? ""))</th><td>\(escape(text(entry, "text") ?? ""))</td></tr>")
        }
        body += "<div class=\"table-wrap\" tabindex=\"0\" role=\"region\" aria-label=\"where it stands\"><table><tbody>\n\(rows.joined(separator: "\n"))\n</tbody></table></div>\n</section>\n"
    }
    return body
}

func page(runs: [Object], manifest: Object, tries: [Object], css: String) -> String {
    let passing = runs.filter(passed)
    let newest = passing.last
    let failedCount = runs.count - passing.count
    var body = ""

    // header
    body += "<h1>performance</h1>\n"
    body += "<p class=\"lead\">every golden run, measured end to end. the numbers on this page come from the log, and nothing here is a goal.</p>\n"
    if runs.isEmpty {
        body += "<p class=\"status\">no runs yet. run <code>scripts/golden.sh</code> to write the first line of <code>docs/performance.jsonl</code>.</p>\n"
    } else {
        var summary = "\(runs.count) \(runs.count == 1 ? "run" : "runs") logged, \(passing.count) passed"
        if failedCount > 0 { summary += ", \(failedCount) failed" }
        if let newest { summary += ". newest passing: \(escape(date(newest))), \(escape(label(newest)))" }
        body += "<p class=\"status\">\(summary)</p>\n"
    }

    // scoreboard
    body += "<section aria-labelledby=\"scoreboard\"><h2 id=\"scoreboard\">scoreboard</h2>\n"
    if let run = newest {
        body += "<p class=\"dim\">the newest passing run: \(escape(date(run))), \"\(escape(label(run)))\". speed is audio seconds per wall second, from the median of the warm repeats. a failed run is never shown here.</p>\n"
        func clip(_ name: String) -> Object? { (value(run, "clips", name) as? Object) }
        func cells(_ render: (Object?) -> String) -> [String] { clipNames.map { render(clip($0)) } }
        var rows: [String] = []
        rows.append(row("audio", cells { seconds(number($0, "seconds")) }))
        rows.append(row("wall time, warm", cells { seconds(number($0, "warm_wall")) }))
        rows.append(row("wall time, cold", cells { seconds(number($0, "cold_wall")) }))
        rows.append(row("speed", cells { times(number($0, "speed")) }))
        rows.append(row("tokens", cells { whole(number($0, "tokens")) }))
        rows.append(row("tokens per second", cells { fixed(number($0, "tokens_per_second"), 1) }))
        rows.append(row("decoder steps", cells { whole(number($0, "steps")) }))
        rows.append(row("decoder steps per second", cells { fixed(number($0, "decoder_steps_per_second"), 1) }))
        rows.append(row("time to first text", cells { seconds(number($0, "time_to_first_text")) }))
        rows.append(row("mel", cells { seconds(number($0, "mel")) }))
        rows.append(row("encode", cells { seconds(number($0, "encode")) }))
        rows.append(row("prefill", cells { seconds(number($0, "prefill")) }))
        rows.append(row("decode", cells { seconds(number($0, "decode")) }))
        rows.append(row("safety margin used", cells { percent(number($0, "margin_used")) }))
        body += table("scoreboard by clip", head: ["per clip"] + clipNames, rows: rows)
        body += "\n<p class=\"small dim\">time in mel, encode, prefill and decode is the best warm repeat. margin used is the share of the safety margin the closest beam comparison took: it must stay under 100%.</p>\n"

        let wide: [String] = [
            row("batch throughput", [times(number(run, "batch_throughput"))]),
            row("sustained throughput", [times(number(run, "sustained_throughput"))]),
            row("cold load", [seconds(number(run, "cold_load"))]),
            row("warm load", [seconds(number(run, "warm_load"))]),
            row("peak memory", [gigabytes(number(run, "peak_memory_bytes"))]),
            row("thermal state", ["<span>\(thermal(number(run, "thermal_state")))</span>"]),
            row("placement", ["<span>\(escape(placement(run)))</span>"]),
            row("golden test", [mark(passed(run))]),
        ]
        body += table("scoreboard for the whole run", head: ["whole run", "value"], rows: wide)
        body += "\n<p class=\"small dim\">batch throughput is the three clips decoded together. sustained throughput is the three clips three times over at the widest batch. both are audio seconds per wall second.</p>\n"
    } else if let last = runs.last {
        body += "<p class=\"dim\">no run has passed yet, so there is no scoreboard. the newest run, \"\(escape(label(last)))\", failed:</p>\n<ul>"
        for failure in (last["failures"] as? [String]) ?? [] { body += "<li>\(escape(failure.lowercased()))</li>" }
        body += "</ul>\n"
    } else {
        body += "<p class=\"dim\">no runs yet.</p>\n"
    }
    body += "</section>\n"

    // baselines
    body += "<section aria-labelledby=\"baselines\"><h2 id=\"baselines\">baselines</h2>\n"
    let reference = manifest["reference"] as? Object
    let versions = [text(reference, "transformers").map { "transformers \($0)" }, text(reference, "torch").map { "torch \($0)" }]
        .compactMap { $0 }.joined(separator: ", ")
    let threads = text(reference, "threads").map { ", \($0) threads" } ?? ""
    let device = [text(reference, "device"), text(reference, "dtype")].compactMap { $0 }.joined(separator: ", ")
    if manifest.isEmpty {
        body += "<p class=\"dim\">golden/manifest.json could not be read, so there are no baselines.</p>\n"
    } else {
        body += "<p class=\"dim\">the reference is hugging face transformers on this mac (\(escape(versions))), \(escape(device))\(escape(threads)), a batch of one. its speed is the clip's audio length divided by the <code>reference_seconds</code> recorded when the golden set was made. the 8-bit mlx build and the q8 gguf are a lower-precision class and are not measured here.</p>\n"
        var rows: [String] = []
        let refSeconds: [Double?] = clipNames.map { number(manifest, "clips", $0, "reference_seconds") }
        let audio: [Double?] = clipNames.map { number(manifest, "clips", $0, "seconds") }
        let refSpeed: [Double?] = zip(audio, refSeconds).map { a, r in (a != nil && r != nil && r! > 0) ? a! / r! : nil }
        let ours: [Double?] = clipNames.map { name in newest.flatMap { number($0, "clips", name, "speed") } }
        rows.append(row("audio", audio.map(seconds)))
        rows.append(row("transformers, wall time", refSeconds.map(seconds)))
        rows.append(row("transformers, speed", refSpeed.map(times)))
        rows.append(row("fourierkit, speed", ours.map(times)))
        rows.append(row("fourierkit against transformers", zip(ours, refSpeed).map { o, r in
            (o != nil && r != nil && r! > 0) ? times(o! / r!) : none
        }))
        body += table("transformers against fourierkit", head: ["per clip"] + clipNames, rows: rows)
        body += "\n<p class=\"small dim\">the last row is how many times faster than transformers the newest passing run is. under 1.00x means slower.</p>\n"
    }
    body += "</section>\n"

    // the climb
    body += "<section aria-labelledby=\"climb\"><h2 id=\"climb\">the climb</h2>\n"
    if runs.isEmpty {
        body += "<p class=\"dim\">no runs yet.</p>\n"
    } else {
        body += "<p class=\"dim\">speed on the long clip and sustained throughput, run by run in the order they were logged. a filled point passed the golden test. a hollow point failed it. the numbers under the chart name each run.</p>\n"
        body += "<div class=\"legend\"><span><i class=\"sw\"></i>long clip speed</span><span><i class=\"sw dots\"></i>sustained throughput</span></div>\n"
        let long = runs.map { number($0, "clips", "long", "speed") }
        let sustained = runs.map { number($0, "sustained_throughput") }
        body += chart(
            series: [Series(name: "long clip speed", cssClass: "s1", values: long),
                     Series(name: "sustained throughput", cssClass: "s2", values: sustained)],
            passes: runs.map(passed), title: "long clip speed and sustained throughput by run, in x realtime")
        var rows: [String] = []
        for (index, run) in runs.enumerated() {
            rows.append("<tr><th scope=\"row\" class=\"num\">\(index + 1)</th><td>\(escape(date(run)))</td><td>\(escape(label(run)))</td>"
                + "<td class=\"num\">\(times(number(run, "clips", "long", "speed")))</td>"
                + "<td class=\"num\">\(times(number(run, "batch_throughput")))</td>"
                + "<td class=\"num\">\(times(number(run, "sustained_throughput")))</td>"
                + "<td>\(mark(passed(run)))</td></tr>")
        }
        let heads = ["run", "date", "label", "long speed", "batch throughput", "sustained", "golden test"]
        let headCells = heads.enumerated().map { index, name in
            "<th scope=\"col\"\(index == 0 || (3...5).contains(index) ? " class=\"num\"" : "")>\(name)</th>"
        }.joined()
        body += "<div class=\"table-wrap\" tabindex=\"0\" role=\"region\" aria-label=\"every logged run\"><table><thead><tr>\(headCells)</tr></thead><tbody>\n\(rows.joined(separator: "\n"))\n</tbody></table></div>\n"
        let failures = runs.enumerated().filter { !passed($0.element) }
        if !failures.isEmpty {
            body += "<h3 style=\"margin-top:32px\">why runs failed</h3>\n<ul class=\"rules\">\n"
            for (index, run) in failures {
                let reasons = ((run["failures"] as? [String]) ?? []).map { escape($0.lowercased()) }
                body += "<li>run \(index + 1): \(reasons.isEmpty ? "no reason was logged" : reasons.joined(separator: "; "))</li>\n"
            }
            body += "</ul>\n"
        }
    }
    body += "</section>\n"

    // method
    body += """
    <section aria-labelledby="method"><h2 id="method">method</h2>
    <p class="dim">how a number gets onto this page.</p>
    <ol class="rules">
    <li>one test produces every line: <code>swift test -c release</code>, the golden test. it needs the model prepared and the golden tensors. there are no other tests.</li>
    <li>the reference is hugging face transformers at one pinned commit, running the pinned checkpoint on the cpu in 32-bit float with a batch of one. it is frozen in <code>golden/</code>.</li>
    <li>a run passes only if all three clips give the reference's token ids and text, stop at the same token, give bit-identical scores on repeat, alone, in a wide batch, after a slot refill and after a reload, and keep every beam comparison inside its safety margin. a run that fails any of these is logged as failed and never counts as a result.</li>
    <li>each clip runs five times. the first is the cold run. the median of the four warm runs is the wall time, and speed is the clip's audio length divided by it. time in mel, encode, prefill and decode is the best warm run.</li>
    <li>release builds, a quiet machine and a nominal thermal state. the thermal state is logged with every run. this mac has no fan, so long runs can slow down as it heats up, which is why sustained throughput is recorded apart from the single-clip speeds.</li>
    <li>a tie is within 3%. only a change that is exact and beats noise is kept.</li>
    <li>timing runs go one at a time. the machine's state is in each run's label when it was not quiet.</li>
    </ol>
    <p class="small dim">the log is <code>docs/performance.jsonl</code>, one json line per run. this page is rebuilt from it with <code>swift scripts/performance.swift</code>.</p>
    </section>

    """

    // tries
    body += "<section aria-labelledby=\"tries\"><h2 id=\"tries\">tries</h2>\n"
    if tries.isEmpty {
        body += "<p class=\"dim\">no tries are logged yet. each experiment goes in <code>docs/tries.jsonl</code> as one json line, kept or dropped.</p>\n"
    } else {
        let kept = tries.filter { (text($0, "verdict") ?? "").lowercased() == "kept" }.count
        let dropped = tries.filter { (text($0, "verdict") ?? "").lowercased() == "dropped" }.count
        body += "<p class=\"dim\">\(tries.count) \(tries.count == 1 ? "try" : "tries") logged: \(kept) kept, \(dropped) dropped. a dropped try is information too.</p>\n"
        var rows: [String] = []
        for entry in tries {
            let verdict = (text(entry, "verdict") ?? none).lowercased()
            let tone = verdict == "kept" ? "pass" : "fail"
            func cell(_ key: String) -> String { escape(text(entry, key) ?? none) }
            rows.append("<tr><td class=\"num\">\(cell("date"))</td><td>\(cell("bet"))</td><td>\(cell("change"))</td><td>\(cell("result"))</td>"
                + "<td><span class=\"mark \(tone)\">\(escape(verdict))</span></td><td>\(cell("why"))</td></tr>")
        }
        let heads = ["date", "bet", "change", "result", "verdict", "why"]
        let headCells = heads.enumerated().map { index, name in
            "<th scope=\"col\"\(index == 0 ? " class=\"num\"" : "")>\(name)</th>"
        }.joined()
        body += "<div class=\"table-wrap\" tabindex=\"0\" role=\"region\" aria-label=\"every logged try\"><table><thead><tr>\(headCells)</tr></thead><tbody>\n\(rows.joined(separator: "\n"))\n</tbody></table></div>\n"
    }
    body += "</section>\n"

    return """
    <!doctype html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
    <meta name="theme-color" content="#000000">
    <title>fourier: performance</title>
    <link rel="preconnect" href="https://cdn.karanganesan.com" crossorigin>
    <link rel="preload" href="https://cdn.karanganesan.com/1794861360796-04033bdf1852/fonts/allo/AlloVariable.woff2" as="font" type="font/woff2" crossorigin>
    <link rel="stylesheet" href="https://cdn.karanganesan.com/1794861360796-04033bdf1852/fonts/allo/allo.css">
    <style>
    \(css)</style>
    </head>
    <body>
    <nav class="top" aria-label="docs">
      <a href="plan.html">plan</a>
      <a href="architecture.html">architecture</a>
      <a href="api.html">api</a>
      <a href="performance.html" aria-current="page">performance</a>
    </nav>
    <main>
    \(body)
    <footer>built from docs/performance.jsonl by scripts/performance.swift.</footer>
    </main>
    </body>
    </html>

    """
}

let pageCSS = #"""
:root {
  color-scheme: dark;
  --w8: rgb(255 255 255 / .08);
  --w16: rgb(255 255 255 / .16);
  --w24: rgb(255 255 255 / .24);
  --w56: rgb(255 255 255 / .56);
  --w64: rgb(255 255 255 / .64);
  --npu: #fff;
  --gpu: rgb(255 255 255 / .56);
  --cpu: rgb(255 255 255 / .16);
  --ease: cubic-bezier(.23, 1, .32, 1);
  font-family: Allo, -apple-system, BlinkMacSystemFont, sans-serif;
  font-optical-sizing: auto;
}
* { box-sizing: border-box; }
html { background: #000; -webkit-text-size-adjust: 100%; scroll-padding-top: 80px; }
body {
  margin: 0;
  background: #000;
  color: #fff;
  font-size: 16px;
  line-height: 24px;
  -webkit-font-smoothing: antialiased;
  -webkit-tap-highlight-color: transparent;
}
a { color: inherit; text-decoration: none; }
@media (hover: hover) and (pointer: fine) { a:hover { text-decoration: underline; } }
:focus-visible { outline: 2px solid rgb(255 255 255 / .48); outline-offset: 2px; }
.top {
  position: sticky;
  top: 0;
  z-index: 8;
  display: flex;
  gap: 8px;
  padding: 16px max(24px, env(safe-area-inset-left));
  background: rgb(0 0 0 / .8);
  -webkit-backdrop-filter: blur(16px);
  backdrop-filter: blur(16px);
}
.top a {
  display: inline-flex;
  align-items: center;
  height: 40px;
  padding: 0 16px;
  border-radius: 999px;
  color: var(--w64);
  font-size: 14px;
}
.top a[aria-current] { background: var(--w16); color: #fff; }
main { width: min(100%, 1072px); margin: 0 auto; padding: 48px 24px 96px; }
h1, h2, h3 { font-weight: 600; text-wrap: balance; margin: 0; }
h1 { font-size: 48px; line-height: 56px; letter-spacing: -.032em; max-width: 800px; }
h2 { font-size: 32px; line-height: 40px; letter-spacing: -.024em; margin-top: 96px; }
h3 { font-size: 16px; line-height: 24px; }
p, li { text-wrap: pretty; }
p { margin: 16px 0 0; max-width: 720px; }
.lead { font-size: 24px; line-height: 32px; letter-spacing: -.012em; color: var(--w64); margin-top: 24px; }
.dim { color: var(--w64); }
.small { font-size: 14px; line-height: 24px; }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 14px; }
code { padding: 2px 4px; border-radius: 4px; background: var(--w8); }
pre { margin: 24px 0 0; padding: 24px; overflow-x: auto; border-radius: 16px; background: var(--w8); line-height: 24px; }
pre code { padding: 0; background: none; }
.panel, pre, .table-wrap, .flow-group, .way, .tile i, .lane, .status { border-radius: 16px; }
@supports (corner-shape: superellipse(1.74)) {
  .panel, pre, .table-wrap, .flow-group, .way, .lane, .status { corner-shape: superellipse(1.74); border-radius: 24.46px; }
  .tile i, code { corner-shape: superellipse(1.74); border-radius: 6.11px; }
  .top a, button { corner-shape: round; }
}
.status { display: inline-block; margin-top: 24px; padding: 8px 16px; background: var(--w8); color: var(--w64); font-size: 14px; }
.legend { display: flex; flex-wrap: wrap; gap: 8px 24px; margin-top: 32px; font-size: 14px; color: var(--w64); }
.legend span { display: inline-flex; align-items: center; gap: 8px; }
.legend b { width: 16px; height: 16px; border-radius: 4px; }
.n { background: var(--npu); } .g { background: var(--gpu); } .c { background: var(--cpu); }

.flow { display: grid; grid-template-columns: 1fr 24px 1fr 24px 1.32fr 24px 1fr; align-items: stretch; gap: 8px; margin-top: 32px; }
.flow-group { padding: 24px; background: var(--w8); }
.flow-group h3 { margin-bottom: 8px; }
.flow-group ol { list-style: none; margin: 16px 0 0; padding: 0; display: grid; gap: 8px; font-size: 14px; }
.flow-group li { display: flex; align-items: center; gap: 8px; }
.flow-group li b { flex: none; display: inline-flex; gap: 2px; width: 32px; }
.flow-group li b i { width: 8px; height: 8px; border-radius: 2px; }
.flow-group .x { opacity: 0; }
.arrow { align-self: center; justify-self: center; color: var(--w56); }
.repeat { display: inline-flex; align-items: center; gap: 8px; color: var(--w64); font-size: 14px; }

.table-wrap { margin-top: 24px; overflow-x: auto; background: var(--w8); }
table { width: 100%; border-collapse: collapse; font-size: 14px; line-height: 24px; }
th, td { padding: 12px 16px; text-align: left; vertical-align: top; }
thead th { color: var(--w64); font-weight: 500; white-space: nowrap; }
tbody tr:nth-child(odd) td, tbody tr:nth-child(odd) th { background: rgb(255 255 255 / .04); }
.steps th[scope=row] { min-width: 288px; font-weight: 400; }
.steps .num { display: inline-block; width: 24px; color: var(--w56); font-variant-numeric: tabular-nums; }
.steps strong { font-weight: 600; }
.steps small { display: block; margin: 4px 0 0 24px; color: var(--w64); font-size: 14px; line-height: 24px; }
.group-row th { background: var(--w16) !important; font-weight: 600; }
.chipcell { width: 88px; text-align: center; vertical-align: middle; }
.mark { display: inline-flex; align-items: center; justify-content: center; min-width: 64px; height: 24px; padding: 0 8px; border-radius: 999px; font-size: 12px; line-height: 16px; }
.race-n { background: var(--npu); color: #000; }
.race-g { background: var(--gpu); color: #000; }
.run-c { background: var(--w24); color: #fff; }
.back { color: var(--w56); }
.none { color: var(--w24); }

.switch { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 24px; }
button {
  height: 40px;
  padding: 0 16px;
  border: 0;
  border-radius: 999px;
  background: var(--w8);
  color: var(--w64);
  font: inherit;
  font-size: 14px;
  cursor: pointer;
  transition: transform 120ms var(--ease), background-color 120ms var(--ease), color 120ms var(--ease);
}
button:active { transform: scale(.97); }
button[aria-pressed=true] { background: #fff; color: #000; }
@media (hover: hover) and (pointer: fine) { button[aria-pressed=false]:hover { background: var(--w16); color: #fff; } }
.note { min-height: 48px; margin-top: 16px; color: var(--w64); }
.packer { display: grid; grid-template-columns: 64px minmax(0, 1fr); gap: 8px; margin-top: 16px; }
.lanes { position: relative; height: 384px; }
.lane-labels { position: relative; height: 384px; font-size: 14px; color: var(--w64); }
.lane-labels span { position: absolute; left: 0; display: flex; align-items: center; }
.lane { position: absolute; left: 0; right: 0; background: var(--w8); }
.free { display: flex; align-items: center; justify-content: center; color: var(--w56); font-size: 12px; }
.tile {
  position: absolute;
  top: 0;
  left: 0;
  width: calc(100% / 12);
  height: 24px;
  padding: 2px;
  transition: transform 320ms var(--ease), opacity 160ms var(--ease);
  will-change: transform;
}
.tile i {
  display: flex;
  width: 100%;
  height: 100%;
  align-items: center;
  justify-content: center;
  border-radius: 4px;
  font-size: 12px;
  font-style: normal;
  line-height: 16px;
  font-variant-numeric: tabular-nums;
  transition: background-color 160ms var(--ease), color 160ms var(--ease);
}
.tile.hidden { opacity: 0; }
.stat { margin-bottom: 8px; font-size: 32px; line-height: 40px; font-weight: 600; letter-spacing: -.024em; font-variant-numeric: tabular-nums; }
.tile.on-n i { background: var(--npu); color: #000; }
.tile.on-g i { background: var(--gpu); color: #000; }
.tile.on-c i { background: var(--w24); color: #fff; }

.ways { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 24px; }
.way { padding: 24px; background: var(--w8); }
.way p { margin-top: 8px; font-size: 14px; color: var(--w64); }
.mini { display: grid; gap: 2px; margin-bottom: 16px; height: 64px; grid-template-columns: repeat(8, 1fr); grid-template-rows: repeat(3, 1fr); }
.mini i { border-radius: 2px; background: rgb(255 255 255 / .04); }
.mini .n { background: var(--npu); } .mini .g { background: var(--gpu); } .mini .c { background: var(--w24); }

.bars { display: grid; gap: 16px; margin-top: 24px; padding: 24px; background: var(--w8); }
.bar-row { display: grid; grid-template-columns: 192px minmax(0, 1fr) 128px; align-items: center; gap: 16px; font-size: 14px; font-variant-numeric: tabular-nums; }
.bar { height: 24px; border-radius: 4px; background: var(--gpu); transform-origin: left; }
.bar-row span:last-child { text-align: right; color: var(--w64); }

ol.rules { margin: 24px 0 0; padding-left: 24px; max-width: 720px; }
ol.rules li + li { margin-top: 8px; }
footer { margin-top: 96px; color: var(--w56); font-size: 14px; }

/* plan page */
.prose h2 { scroll-margin-top: 80px; }
.prose ul, .prose ol { margin: 16px 0 0; padding-left: 24px; max-width: 720px; }
.prose li + li { margin-top: 8px; }
.prose li > ul, .prose li > ol { margin-top: 8px; }
.prose strong { font-weight: 600; }
.prose table code { white-space: nowrap; }
.toc { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 32px; font-size: 14px; }
.toc a { display: inline-flex; align-items: center; height: 40px; padding: 0 16px; border-radius: 999px; background: var(--w8); color: var(--w64); }

@media (max-width: 880px) {
  h1 { font-size: 32px; line-height: 40px; }
  h2 { font-size: 24px; line-height: 32px; margin-top: 64px; }
  .lead { font-size: 16px; line-height: 24px; }
  main { padding-top: 24px; }
  .flow { grid-template-columns: 1fr; }
  .arrow { transform: rotate(90deg); }
  .ways { grid-template-columns: 1fr; }
  .packer { grid-template-columns: 40px minmax(0, 1fr); }
  .tile i { font-size: 8px; }
  .bar-row { grid-template-columns: 120px minmax(0, 1fr) 72px; gap: 8px; }
  .steps th[scope=row] { min-width: 224px; }
  button { height: 48px; }
}
@media (prefers-reduced-motion: reduce) {
  .tile { transition: opacity 160ms ease; }
  button { transition: none; }
  button:active { transform: none; }
}

/* api and performance pages */
.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
thead th.num { text-align: right; }
td.mono, th.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 14px; }
.mark.pass { background: #fff; color: #000; }
.mark.fail { background: var(--w24); color: #fff; }
.chart { margin-top: 24px; padding: 24px; background: var(--w8); border-radius: 16px; }
@supports (corner-shape: superellipse(1.74)) { .chart { corner-shape: superellipse(1.74); border-radius: 24.46px; } }
.chart { overflow-x: auto; }
.chart svg { display: block; width: 100%; min-width: 640px; height: auto; }
.chart text { fill: rgb(255 255 255 / .64); font-family: inherit; font-size: 12px; font-variant-numeric: tabular-nums; }
.chart .grid { stroke: rgb(255 255 255 / .08); stroke-width: 1; }
.chart .s1 { stroke: #fff; fill: none; stroke-width: 2; }
.chart .s2 { stroke: rgb(255 255 255 / .56); fill: none; stroke-width: 2; stroke-dasharray: 2 6; stroke-linecap: round; }
.chart .p1 { stroke: #fff; stroke-width: 2; }
.chart .p2 { stroke: rgb(255 255 255 / .56); stroke-width: 2; }
.chart .full { fill: #fff; }
.chart .full.p2 { fill: rgb(255 255 255 / .56); }
.chart .hollow { fill: #000; }
.sw { display: inline-block; width: 16px; height: 2px; vertical-align: middle; background: #fff; }
.sw.dots { background: none; border-top: 2px dotted rgb(255 255 255 / .56); height: 0; }
.api-toc { margin-top: 32px; }
.sig { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 14px; }
td .sig, th .sig { white-space: nowrap; }
@media (max-width: 880px) { .chart { padding: 16px; } }
"""#

let lines = readLines(root.appendingPathComponent("docs/performance.jsonl"))
// engine runs feed the scoreboard and the climb. lines from the modes test hold the live numbers.
let runs = lines.filter { text($0, "kind") != "modes" }
let liveRuns = lines.filter { text($0, "kind") == "modes" }
let standing = readLines(root.appendingPathComponent("docs/standing.jsonl"))
let manifest = readObject(root.appendingPathComponent("golden/manifest.json"))
let tries = readLines(root.appendingPathComponent("docs/tries.jsonl"))
var html = page(runs: runs, manifest: manifest, tries: tries, css: pageCSS)
html = html.replacingOccurrences(of: "<section aria-labelledby=\"tries\">", with: extraSections(live: liveRuns.last, standing: standing) + "<section aria-labelledby=\"tries\">")
let target = root.appendingPathComponent("docs/performance.html")
do {
    try html.write(to: target, atomically: true, encoding: .utf8)
    print("wrote docs/performance.html from \(runs.count) \(runs.count == 1 ? "run" : "runs") and \(tries.count) \(tries.count == 1 ? "try" : "tries")")
} catch {
    FileHandle.standardError.write(Data("could not write docs/performance.html: \(error)\n".utf8))
    exit(1)
}
