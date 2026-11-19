import json
import os
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_ID = r"(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?"


@dataclass
class Source:
    title: str
    url: str
    authors: list = field(default_factory=list)
    year: str = ""
    arxiv_id: str = ""
    doi: str = ""
    passages: list = field(default_factory=list)


def clean(text):
    return " ".join(text.split())


def get(url, limit=8_000_000):
    request = Request(url, headers={"User-Agent": "LocalBilingualResearchAssistant/0.1"})
    with urlopen(request, timeout=12) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError("Source response exceeded the size limit.")
    return data


def normalize_arxiv(value):
    if value.startswith(("https://arxiv.org/", "http://arxiv.org/")):
        value = urlparse(value).path.removeprefix("/abs/").removeprefix("/pdf/")
        value = value.removesuffix(".pdf")
    if not re.fullmatch(ARXIV_ID, value):
        raise ValueError(f"Invalid arXiv ID: {value}")
    return value


def arxiv_search(terms, ids, limit):
    params = {"max_results": limit}
    if ids:
        params["id_list"] = ",".join(ids)
    else:
        phrases = [clean(re.sub(r'[^\w -]', ' ', term)) for term in terms]
        phrases = [phrase for phrase in phrases if phrase]
        if not phrases:
            return []
        params["search_query"] = " OR ".join('all:"' + phrase + '"' for phrase in phrases)
        params["sortBy"] = "relevance"
    feed = ET.fromstring(get("https://export.arxiv.org/api/query?" + urlencode(params)))
    sources = []
    for entry in feed.findall(ATOM + "entry"):
        if "/abs/" not in entry.findtext(ATOM + "id", ""):
            continue
        identifier = entry.findtext(ATOM + "id").split("/abs/", 1)[1]
        if not re.fullmatch(ARXIV_ID, identifier):
            continue
        abstract = clean(entry.findtext(ATOM + "summary", ""))
        sources.append(Source(
            title=clean(entry.findtext(ATOM + "title", "")),
            url=f"https://arxiv.org/abs/{identifier}",
            authors=[clean(author.findtext(ATOM + "name", "")) for author in entry.findall(ATOM + "author")],
            year=entry.findtext(ATOM + "published", "")[:4],
            arxiv_id=identifier,
            doi=entry.findtext("{http://arxiv.org/schemas/atom}doi", ""),
            passages=[{"section": "Abstract", "text": abstract}] if abstract else [],
        ))
    return sources


class ArticleParagraphs(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paragraphs = []
        self.stack = []
        self.paragraph = None
        self.heading = None
        self.section = "Article text"

    def handle_starttag(self, tag, attrs):
        if tag in {"area", "base", "br", "col", "embed", "hr", "img", "input",
                   "link", "meta", "param", "source", "track", "wbr"}:
            return
        classes = set(dict(attrs).get("class", "").split())
        self.stack.append((tag, classes))
        if tag in {"h2", "h3", "h4", "h5"} and "ltx_title" in classes:
            self.heading = []
        if "ltx_p" in classes and self.paragraph is None:
            self.paragraph = []
        if tag == "math" and self.paragraph is not None:
            self.paragraph.append(dict(attrs).get("alttext", ""))

    def handle_endtag(self, tag):
        matching = next((index for index in range(len(self.stack) - 1, -1, -1)
                         if self.stack[index][0] == tag), None)
        if matching is None:
            return
        popped = self.stack[matching:]
        del self.stack[matching:]
        if self.heading is not None and any(item[0] in {"h2", "h3", "h4", "h5"} for item in popped):
            self.section = clean(" ".join(self.heading))
            self.heading = None
        if self.paragraph is not None and any("ltx_p" in item[1] for item in popped):
            text = clean("".join(self.paragraph))
            self.paragraph = None
            excluded = ("reference", "acknowledg", "bibliograph", "abstract")
            if len(text) >= 100 and not any(word in self.section.lower() for word in excluded):
                self.paragraphs.append({"section": self.section, "text": text})

    def handle_data(self, data):
        if self.heading is not None:
            self.heading.append(data)
        if self.paragraph is not None:
            self.paragraph.append(data)


def openalex_search(terms, limit):
    params = {
        "search": " ".join(terms),
        "per_page": limit,
        "select": "id,doi,title,publication_year,abstract_inverted_index,authorships",
    }
    key = os.environ.get("OPENALEX_API_KEY")
    if key:
        params["api_key"] = key
    records = json.loads(get("https://api.openalex.org/works?" + urlencode(params)))["results"]
    sources = []
    for record in records:
        index = record.get("abstract_inverted_index") or {}
        words = sorted((pos, word) for word, positions in index.items() for pos in positions)
        if not words:
            continue
        doi_url = record.get("doi") or ""
        abstract = " ".join(word for _, word in words)
        sources.append(Source(
            title=record.get("title") or "(untitled)",
            url=doi_url or record["id"],
            authors=[item["author"]["display_name"] for item in record.get("authorships", [])
                     if (item.get("author") or {}).get("display_name")],
            year=str(record.get("publication_year") or ""),
            doi=doi_url.removeprefix("https://doi.org/"),
            passages=[{"section": "Abstract", "text": abstract}],
        ))
    return sources


def add_arxiv_passages(source, terms):
    url = f"https://arxiv.org/html/{source.arxiv_id}"
    try:
        parser = ArticleParagraphs()
        parser.feed(get(url).decode("utf-8", errors="replace"))
    except (OSError, ValueError):
        return source
    significant = set(re.findall(r"[a-z]{4,}", " ".join(terms).lower()))
    candidates = sorted(
        parser.paragraphs,
        key=lambda p: (
            len(significant.intersection(re.findall(r"[a-z]{4,}", p["text"].lower()))),
            any(word in p["section"].lower() for word in ("result", "discussion", "limitation", "conclusion")),
        ),
        reverse=True,
    )
    seen_sections = set()
    for passage in candidates:
        if passage["section"] in seen_sections:
            continue
        seen_sections.add(passage["section"])
        source.passages.append(dict(passage, url=url))
        if len(seen_sections) == 3:
            break
    return source


def collect_sources(terms, ids, limit, local_file=None):
    if local_file is not None:
        records = json.loads(local_file.read_text(encoding="utf-8"))
        if not isinstance(records, list) or not records:
            raise ValueError("The local source file must be a nonempty JSON array.")
        try:
            sources = [Source(**record) for record in records[:limit]]
            for source in sources:
                if not (isinstance(source.title, str) and isinstance(source.url, str)):
                    raise ValueError("Every source needs a title and HTTPS URL.")
                if not isinstance(source.authors, list) or any(not isinstance(a, str) for a in source.authors):
                    raise ValueError("Source authors must be an array of names.")
                if not isinstance(source.passages, list) or any(
                    not isinstance(p, dict) or not isinstance(p.get("text"), str)
                    or not isinstance(p.get("section"), str) for p in source.passages
                ):
                    raise ValueError("Every passage needs text and a section name.")
                source.year = str(source.year or "")
        except TypeError as exc:
            raise ValueError("The local sources must follow the layout in examples/sources.json.") from exc
        return sources, []
    issues = []
    sources = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {"arXiv": pool.submit(arxiv_search, terms, ids, limit)}
        if not ids:
            pending["OpenAlex"] = pool.submit(openalex_search, terms[:1], limit)
        results = {}
        for provider, job in pending.items():
            try:
                results[provider] = job.result()
            except (OSError, ValueError, ET.ParseError) as exc:
                # Keep URL query parameters out of logs.
                issues.append(f"{provider} unavailable ({type(exc).__name__})")
        from_arxiv = results.get("arXiv", [])
        from_openalex = results.get("OpenAlex", [])
        source_keys = set()
        # Reserve space for both indexes when possible.
        for record in [*from_arxiv[:1], *from_openalex[:1], *from_arxiv[1:], *from_openalex[1:]]:
            key = re.sub(r"[^\w]+", "", record.title.casefold())
            if key in source_keys or (not record.passages and not record.arxiv_id):
                continue
            sources.append(record)
            source_keys.add(key)
            if len(sources) == limit:
                break
        sources = list(pool.map(
            lambda source: add_arxiv_passages(source, terms) if source.arxiv_id else source,
            sources,
        ))
    if ids:
        found = {re.sub(r"v\d+$", "", item.arxiv_id) for item in sources}
        for identifier in ids:
            if re.sub(r"v\d+$", "", identifier) not in found:
                issues.append(f"arXiv record unavailable: {identifier}")
    return sources, issues


def excerpt(text, limit=1100):
    text = clean(text)
    if len(text) <= limit:
        return text
    endings = list(re.finditer(r"[.!?](?=\s+[A-Z])", text[:limit]))
    if endings and endings[-1].end() >= limit // 2:
        return text[:endings[-1].end()]
    end = text.rfind(" ", 0, limit)
    return text[:end if end > 0 else limit]
