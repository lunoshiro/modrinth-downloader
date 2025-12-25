#!/usr/bin/env python3
"""Modrinth Downloader"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import socket
import sys
import time
import zipfile
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union
from urllib import error, request
from urllib.parse import quote_plus

# Version info
__version__ = "1.1.1"
__author__ = "OwnExcept"

# ===========================
# Constants & Configuration
# ===========================

API_BASE = "https://api.modrinth.com"
DEFAULT_USER_AGENT = "ModrinthDownloader/1.x.x (python 3.8+)"
DEFAULT_MAX_WORKERS = os.cpu_count() or 6
DEFAULT_TIMEOUT = 30
DEFAULT_MAX_RETRIES = 3
CSV_FILE = "projectid.csv"
LOG_FILE = "modrinth_dl.log"

# ===========================
# Custom Exceptions
# ===========================

class ModrinthError(Exception):
    """Base exception for all Modrinth-related errors."""
    pass


class NetworkError(ModrinthError):
    """Raised when network connectivity issues occur."""
    pass


class APIError(ModrinthError):
    """Raised when API requests fail."""
    pass


class DownloadError(ModrinthError):
    """Raised when file downloads fail."""
    pass


class ValidationError(ModrinthError):
    """Raised when data validation fails."""
    pass


class ConfigurationError(ModrinthError):
    """Raised when configuration is invalid."""
    pass


# ===========================
# Enums & Type Definitions
# ===========================

class ProjectType(str, Enum):
    """Supported project types on Modrinth."""
    MOD = "mod"
    PLUGIN = "plugin"
    MODPACK = "modpack"
    RESOURCEPACK = "resourcepack"
    SHADER = "shader"
    DATAPACK = "datapack"
    MISC = "misc"

    def get_folder_name(self) -> str:
        """Get the folder name for this project type."""
        return TYPE_FOLDERS.get(self.value, "misc")


TYPE_FOLDERS: Dict[str, str] = {
    ProjectType.MOD.value: "mods",
    ProjectType.PLUGIN.value: "plugins",
    ProjectType.MODPACK.value: "modpacks",
    ProjectType.RESOURCEPACK.value: "resourcepacks",
    ProjectType.SHADER.value: "shaderpacks",
    ProjectType.DATAPACK.value: "datapacks",
}

BACKWARD_COMPAT_TYPES = frozenset({ProjectType.RESOURCEPACK.value, ProjectType.SHADER.value})
PLUGIN_LOADERS = frozenset({"bukkit", "folia", "paper", "purpur", "spigot", "sponge"})
DEFAULT_IGNORE_VERSION_IDS: List[str] = ["WwbubTsV", "sGmHWmeL"]


# ===========================
# Data Models
# ===========================

@dataclass(frozen=True)
class AppConfig:
    """Immutable application configuration."""
    user_agent: str
    max_workers: int
    timeout: int
    max_retries: int
    ignore_version_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.max_workers < 1:
            raise ConfigurationError("max_workers must be at least 1")
        if self.timeout < 1:
            raise ConfigurationError("timeout must be at least 1 second")
        if self.max_retries < 1:
            raise ConfigurationError("max_retries must be at least 1")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> AppConfig:
        """Create config from command line arguments."""
        ignore_ids = tuple(args.ignore_version_ids.split(',')) if args.ignore_version_ids else tuple(DEFAULT_IGNORE_VERSION_IDS)
        return cls(
            user_agent=args.user_agent,
            max_workers=args.workers,
            timeout=args.timeout,
            max_retries=args.max_retries,
            ignore_version_ids=ignore_ids
        )


@dataclass(frozen=True)
class SearchResult:
    """Immutable search result from Modrinth."""
    project_id: str
    slug: str
    title: str
    description: str
    project_type: ProjectType
    downloads: int
    url: str

    def __str__(self) -> str:
        """Format search result for display."""
        return f"{self.title}[{self.slug}] ({self.project_id}): {self.url}"


@dataclass(frozen=True)
class FileInfo:
    """Immutable file information."""
    filename: str
    url: str
    size: int = 0
    hashes: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DownloadTask:
    """Immutable download task specification."""
    project_id: str
    project_type: ProjectType
    file_info: FileInfo
    destination_path: Path
    old_path: Optional[Path] = None


# ===========================
# Protocol Definitions (Interfaces)
# ===========================

class HTTPClient(Protocol):
    """Protocol for HTTP client operations."""
    
    def get(self, url: str, headers: Optional[Dict[str, str]] = None) -> bytes:
        """Perform GET request and return response body."""
        ...
    
    def download_file(self, url: str, destination: Path, headers: Optional[Dict[str, str]] = None) -> bool:
        """Download file to destination with resume support."""
        ...


class ProjectRepository(Protocol):
    """Protocol for project data persistence."""
    
    def save(self, project_id: str, relative_path: str) -> None:
        """Save project mapping."""
        ...
    
    def get(self, project_id: str) -> Optional[str]:
        """Get relative path for project."""
        ...
    
    def get_all(self) -> Dict[str, str]:
        """Get all project mappings."""
        ...
    
    def delete(self, project_id: str) -> None:
        """Delete project mapping."""
        ...


class ModrinthAPIClient(Protocol):
    """Protocol for Modrinth API operations."""
    
    def get_project(self, project_id: str) -> Dict[str, Any]:
        """Get project information."""
        ...
    
    def get_versions(self, project_id: str, loader: Optional[str] = None, 
                    game_version: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get project versions with optional filters."""
        ...
    
    def search_projects(self, query: str, filters: Optional[Dict[str, str]] = None,
                       limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """Search for projects."""
        ...
    
    def get_collection(self, collection_id: str) -> List[str]:
        """Get project IDs from collection."""
        ...


# ===========================
# Concrete Implementations
# ===========================

class StandardHTTPClient:
    """Standard implementation of HTTPClient using urllib."""
    
    def __init__(self, config: AppConfig, logger: logging.Logger) -> None:
        self._config = config
        self._logger = logger
    
    def get(self, url: str, headers: Optional[Dict[str, str]] = None) -> bytes:
        """Perform GET request with retry logic."""
        request_headers = {"User-Agent": self._config.user_agent}
        if headers:
            request_headers.update(headers)
        
        for attempt in range(self._config.max_retries):
            try:
                req = request.Request(url, headers=request_headers)
                with request.urlopen(req, timeout=self._config.timeout) as response:
                    return response.read()
            except (error.URLError, error.HTTPError, OSError) as exc:
                self._logger.warning(
                    f"Request attempt {attempt + 1}/{self._config.max_retries} failed: {exc}",
                    extra={"url": url, "attempt": attempt + 1}
                )
                if attempt < self._config.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise NetworkError(f"Failed to fetch {url} after {self._config.max_retries} attempts") from exc
        
        raise NetworkError(f"Failed to fetch {url}")
    
    def download_file(self, url: str, destination: Path, 
                     headers: Optional[Dict[str, str]] = None) -> bool:
        """Download file with resume support and retry logic."""
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._logger.error(f"Cannot create directory: {exc}")
            raise DownloadError(f"Failed to create directory {destination.parent}") from exc
        
        temp_path = destination.with_suffix(destination.suffix + '.part')
        
        for attempt in range(self._config.max_retries):
            try:
                request_headers = {"User-Agent": self._config.user_agent}
                if headers:
                    request_headers.update(headers)
                
                # Resume support
                if temp_path.exists() and temp_path.stat().st_size > 0:
                    request_headers["Range"] = f"bytes={temp_path.stat().st_size}-"
                    mode = 'ab'
                else:
                    mode = 'wb'
                
                req = request.Request(url, headers=request_headers)
                
                with request.urlopen(req, timeout=self._config.timeout) as response:
                    with open(temp_path, mode) as file:
                        while chunk := response.read(8192):
                            file.write(chunk)
                
                # Move temp to final destination atomically
                if destination.exists():
                    destination.unlink()
                temp_path.rename(destination)
                
                self._logger.info(f"Downloaded: {destination.name}")
                return True
                
            except (error.URLError, error.HTTPError, OSError) as exc:
                self._logger.warning(
                    f"Download attempt {attempt + 1}/{self._config.max_retries} failed: {exc}",
                    extra={"url": url, "attempt": attempt + 1}
                )
                if attempt < self._config.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    # Cleanup partial file
                    if temp_path.exists():
                        try:
                            temp_path.unlink()
                        except OSError:
                            pass
                    raise DownloadError(f"Failed to download {url}") from exc
        
        return False


class CSVProjectRepository:
    """CSV-based implementation of ProjectRepository."""
    
    def __init__(self, directory: Path, logger: logging.Logger) -> None:
        self._csv_path = directory / CSV_FILE
        self._logger = logger
        self._data: Dict[str, str] = self._load()
    
    def _load(self) -> Dict[str, str]:
        """Load project mappings from CSV."""
        if not self._csv_path.exists():
            return {}
        
        try:
            with open(self._csv_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file, delimiter='|')
                
                # Skip and validate header
                try:
                    header = next(reader)
                    if header != ['path', 'id']:
                        self._logger.warning("Invalid CSV header, ignoring file")
                        return {}
                except StopIteration:
                    return {}
                
                return {row[1]: row[0] for row in reader if len(row) == 2}
                
        except (IOError, OSError, UnicodeDecodeError) as exc:
            self._logger.warning(f"Cannot read CSV: {exc}")
            return {}
    
    def save(self, project_id: str, relative_path: str) -> None:
        """Save project mapping."""
        self._data[project_id] = relative_path
        self._persist()
    
    def get(self, project_id: str) -> Optional[str]:
        """Get relative path for project."""
        return self._data.get(project_id)
    
    def get_all(self) -> Dict[str, str]:
        """Get all project mappings."""
        return self._data.copy()
    
    def delete(self, project_id: str) -> None:
        """Delete project mapping."""
        if project_id in self._data:
            del self._data[project_id]
            self._persist()
    
    def _persist(self) -> None:
        """Persist data to CSV file."""
        try:
            with open(self._csv_path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file, delimiter='|')
                writer.writerow(['path', 'id'])
                
                for project_id, relative_path in sorted(self._data.items()):
                    writer.writerow([relative_path, project_id])
                    
        except (IOError, OSError) as exc:
            self._logger.error(f"Cannot write CSV: {exc}")
            raise ModrinthError(f"Failed to persist project data") from exc


class ModrinthAPI:
    """Modrinth API client implementation."""
    
    def __init__(self, http_client: HTTPClient, logger: logging.Logger) -> None:
        self._http = http_client
        self._logger = logger
    
    def get_project(self, project_id: str) -> Dict[str, Any]:
        """Get project information."""
        try:
            data = self._http.get(f"{API_BASE}/v2/project/{project_id}")
            result = json.loads(data.decode('utf-8'))
            
            if not isinstance(result, dict):
                raise APIError(f"Invalid API response for project {project_id}")
            
            return result
        except (json.JSONDecodeError, KeyError) as exc:
            raise APIError(f"Failed to parse project data for {project_id}") from exc
    
    def get_versions(self, project_id: str, loader: Optional[str] = None,
                    game_version: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get project versions with optional filters."""
        query_params = []
        if loader:
            query_params.append(f"loaders=[%22{loader}%22]")
        if game_version:
            query_params.append(f"game_versions=[%22{game_version}%22]")
        
        endpoint = f"/v2/project/{project_id}/version"
        if query_params:
            endpoint += "?" + "&".join(query_params)
        
        try:
            data = self._http.get(f"{API_BASE}{endpoint}")
            result = json.loads(data.decode('utf-8'))
            
            if not isinstance(result, list):
                raise APIError(f"Invalid versions response for {project_id}")
            
            return result
        except (json.JSONDecodeError, KeyError) as exc:
            raise APIError(f"Failed to parse versions for {project_id}") from exc
    
    def search_projects(self, query: str, filters: Optional[Dict[str, str]] = None,
                       limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """Search for projects with optional filters."""
        facets = []
        if filters:
            facet_group = []
            for key, value in filters.items():
                if key == 'loader':
                    facet_group.append(f'categories:{value}')
                elif key == 'version':
                    facet_group.append(f'versions:{value}')
                elif key == 'project_type':
                    facet_group.append(f'project_type:{value}')
            
            if facet_group:
                facets = [[f] for f in facet_group]
        
        params = [
            f"query={quote_plus(query)}",
            f"limit={limit}",
            f"offset={offset}"
        ]
        
        if facets:
            facets_json = json.dumps(facets)
            params.append(f"facets={quote_plus(facets_json)}")
        
        endpoint = f"/v2/search?{'&'.join(params)}"
        
        try:
            data = self._http.get(f"{API_BASE}{endpoint}")
            result = json.loads(data.decode('utf-8'))
            
            if not isinstance(result, dict) or 'hits' not in result:
                raise APIError("Invalid search response")
            
            return result.get('hits', [])
        except (json.JSONDecodeError, KeyError) as exc:
            raise APIError(f"Failed to parse search results") from exc
    
    def get_collection(self, collection_id: str) -> List[str]:
        """Get project IDs from collection."""
        try:
            data = self._http.get(f"{API_BASE}/v3/collection/{collection_id}")
            result = json.loads(data.decode('utf-8'))
            
            if not isinstance(result, dict) or 'projects' not in result:
                raise APIError(f"Invalid collection response for {collection_id}")
            
            return result.get('projects', [])
        except (json.JSONDecodeError, KeyError) as exc:
            raise APIError(f"Failed to parse collection {collection_id}") from exc


# ===========================
# Domain Services
# ===========================

class VersionMatcher:
    """Service for matching compatible versions."""
    
    @staticmethod
    def compare_versions(v1: str, v2: str) -> int:
        """
        Compare Minecraft version strings.
        
        Returns:
            -1 if v1 < v2, 0 if equal, 1 if v1 > v2
        """
        def parse(version: str) -> Tuple[int, ...]:
            try:
                parts = str(version).replace('-', '.').split('.')
                return tuple(int(x) for x in parts if x.isdigit())
            except (ValueError, AttributeError):
                return (0,)
        
        p1, p2 = parse(v1), parse(v2)
        
        if p1 < p2:
            return -1
        elif p1 > p2:
            return 1
        return 0
    
    @classmethod
    def find_best_version(
        cls,
        versions: List[Dict[str, Any]],
        mc_version: str,
        loader: str,
        project_type: ProjectType,
        project_id: str,
        ignore_version_ids: Tuple[str, ...]
    ) -> Optional[Dict[str, Any]]:
        """Find the best compatible version for given criteria."""
        if not versions:
            return None
        
        ignore_ver = project_id in ignore_version_ids
        
        for version in versions:
            if not isinstance(version, dict):
                continue
            
            # Check loader compatibility
            if project_type.value not in BACKWARD_COMPAT_TYPES and project_type != ProjectType.DATAPACK:
                loaders = version.get("loaders", [])
                if loaders and loader not in loaders:
                    continue
            
            # Check MC version compatibility
            if not ignore_ver:
                game_versions = version.get("game_versions", [])
                if not isinstance(game_versions, list):
                    continue
                
                if project_type.value in BACKWARD_COMPAT_TYPES:
                    # Backward compatible: any version <= target
                    compatible = any(
                        cls.compare_versions(gv, mc_version) <= 0 
                        for gv in game_versions
                    )
                    if not compatible:
                        continue
                else:
                    # Exact match required
                    if mc_version not in game_versions:
                        continue
            
            return version
        
        return None


class ModpackExtractor:
    """Service for extracting and processing modpacks."""
    
    def __init__(self, http_client: HTTPClient, config: AppConfig, 
                 logger: logging.Logger) -> None:
        self._http = http_client
        self._config = config
        self._logger = logger
    
    def extract(self, mrpack_path: Path, output_dir: Path) -> bool:
        """Extract modpack and download dependencies."""
        if not mrpack_path.exists():
            self._logger.error(f"Modpack file not found: {mrpack_path}")
            return False
        
        try:
            with zipfile.ZipFile(mrpack_path, 'r') as zf:
                # Read and validate index
                try:
                    with zf.open('modrinth.index.json') as f:
                        data = f.read().decode('utf-8')
                        index = json.loads(data)
                except (KeyError, json.JSONDecodeError) as exc:
                    raise ValidationError("Invalid modpack: missing or corrupt index") from exc
                
                # Extract modpack contents
                extract_path = output_dir / mrpack_path.stem
                extract_path.mkdir(parents=True, exist_ok=True)
                zf.extractall(extract_path)
                
                self._logger.info(f"Extracted modpack to: {extract_path}")
                
                # Download dependencies
                self._download_dependencies(index, extract_path)
                
                return True
                
        except (zipfile.BadZipFile, OSError) as exc:
            self._logger.error(f"Modpack extraction failed: {exc}")
            raise DownloadError(f"Failed to extract modpack {mrpack_path}") from exc
    
    def _download_dependencies(self, index: Dict[str, Any], extract_path: Path) -> None:
        """Download modpack dependencies."""
        files = index.get("files", [])
        if not isinstance(files, list):
            return
        
        for file_info in files:
            if not isinstance(file_info, dict):
                continue
            
            path = file_info.get("path")
            downloads = file_info.get("downloads", [])
            
            if path and isinstance(downloads, list):
                dest = extract_path / path
                for url in downloads:
                    if isinstance(url, str):
                        try:
                            if self._http.download_file(url, dest):
                                break
                        except DownloadError:
                            continue


class SearchService:
    """Service for searching and displaying projects."""
    
    def __init__(self, api: ModrinthAPI, logger: logging.Logger) -> None:
        self._api = api
        self._logger = logger
    
    def search(
        self,
        query: str,
        filters: Optional[Dict[str, str]] = None,
        limit: int = 10
    ) -> List[SearchResult]:
        """Search for projects and return parsed results."""
        try:
            hits = self._api.search_projects(query, filters, limit)
            
            results = []
            for hit in hits:
                if not isinstance(hit, dict):
                    continue
                
                try:
                    result = SearchResult(
                        project_id=hit.get("project_id", ""),
                        slug=hit.get("slug", ""),
                        title=hit.get("title", "Unknown"),
                        description=hit.get("description", ""),
                        project_type=ProjectType(hit.get("project_type", "mod")),
                        downloads=hit.get("downloads", 0),
                        url=f"https://modrinth.com/{hit.get('project_type', 'mod')}/{hit.get('slug', '')}"
                    )
                    results.append(result)
                except (KeyError, TypeError, ValueError) as exc:
                    self._logger.debug(f"Skipping invalid search result: {exc}")
                    continue
            
            return results
            
        except APIError as exc:
            self._logger.error(f"Search failed: {exc}")
            raise
    
    def display_results(self, results: List[SearchResult]) -> None:
        """Display search results in formatted output."""
        if not results:
            print("No results found.")
            return
        
        print(f"\nFound {len(results)} results:\n")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result}")
            if result.description:
                desc = (result.description[:80] + "...") if len(result.description) > 80 else result.description
                print(f"   Description: {desc}")
            print(f"   Downloads: {result.downloads:,}")
            print()


class DownloadService:
    """Service for downloading projects."""
    
    def __init__(
        self,
        api: ModrinthAPI,
        http_client: HTTPClient,
        repository: ProjectRepository,
        version_matcher: VersionMatcher,
        modpack_extractor: ModpackExtractor,
        config: AppConfig,
        logger: logging.Logger
    ) -> None:
        self._api = api
        self._http = http_client
        self._repo = repository
        self._version_matcher = version_matcher
        self._modpack_extractor = modpack_extractor
        self._config = config
        self._logger = logger
    
    def prepare_download_task(
        self,
        project_id: str,
        mc_version: str,
        loader: str,
        output_dir: Path,
        only_type: Optional[str] = None,
        update: bool = False
    ) -> Optional[DownloadTask]:
        """Prepare a download task for a project."""
        try:
            # Get project info
            project_info = self._api.get_project(project_id)
            project_type = ProjectType(project_info.get("project_type", "mod"))
            
            # Filter by type if requested
            if only_type and only_type != project_type.value:
                return None
            
            # Get versions
            versions = self._api.get_versions(project_id, loader, mc_version)
            
            # Find best version
            best_version = self._version_matcher.find_best_version(
                versions, mc_version, loader, project_type, 
                project_id, self._config.ignore_version_ids
            )
            
            if not best_version:
                self._logger.debug(f"No compatible version found for {project_id}")
                return None
            
            # Get primary file
            files = best_version.get("files", [])
            primary_file = next(
                (f for f in files if isinstance(f, dict) and f.get("primary")),
                None
            )
            
            if not primary_file or "filename" not in primary_file or "url" not in primary_file:
                self._logger.warning(f"No valid file found for {project_id}")
                return None
            
            # Determine destination
            if loader.lower() in PLUGIN_LOADERS:
                folder = "plugins" if project_type == ProjectType.MOD else project_type.get_folder_name()
            else:
                folder = project_type.get_folder_name()
            
            destination = output_dir / folder / primary_file["filename"]
            old_path_str = self._repo.get(project_id)
            old_path = Path(old_path_str) if old_path_str else None
            
            # Check if update needed
            if old_path_str and not update:
                return None
            
            if old_path and old_path.name == primary_file["filename"]:
                return None
            
            file_info = FileInfo(
                filename=primary_file["filename"],
                url=primary_file["url"],
                size=primary_file.get("size", 0),
                hashes=primary_file.get("hashes", {})
            )
            
            return DownloadTask(
                project_id=project_id,
                project_type=project_type,
                file_info=file_info,
                destination_path=destination,
                old_path=old_path
            )
            
        except (APIError, ValueError) as exc:
            self._logger.error(f"Failed to prepare download for {project_id}: {exc}")
            return None
    
    def execute_download(self, task: DownloadTask, base_dir: Path) -> bool:
        """Execute a download task."""
        try:
            # Download file
            if not self._http.download_file(task.file_info.url, task.destination_path):
                return False
            
            # Remove old file
            if task.old_path and task.old_path.exists() and task.old_path != task.destination_path:
                try:
                    task.old_path.unlink()
                    self._logger.info(f"Removed old file: {task.old_path}")
                except OSError as exc:
                    self._logger.warning(f"Failed to remove old file: {exc}")
            
            # Update repository
            try:
                relative_path = str(task.destination_path.relative_to(base_dir))
                self._repo.save(task.project_id, relative_path.replace('\\', '/'))
            except ValueError:
                # Fallback to absolute path
                self._repo.save(task.project_id, str(task.destination_path))
            
            # Extract modpack if needed
            if task.project_type == ProjectType.MODPACK and task.destination_path.suffix == '.mrpack':
                try:
                    self._modpack_extractor.extract(
                        task.destination_path,
                        task.destination_path.parent
                    )
                except DownloadError as exc:
                    self._logger.warning(f"Modpack extraction failed: {exc}")
            
            return True
            
        except (DownloadError, OSError) as exc:
            self._logger.error(f"Download failed: {exc}")
            return False


# ===========================
# Application Layer
# ===========================

def setup_logging(directory: Path, level: int = logging.INFO) -> logging.Logger:
    """Configure structured logging with file and console handlers."""
    logger = logging.getLogger("modrinth_dl")
    logger.setLevel(level)
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    try:
        log_path = directory / LOG_FILE
        file_handler = logging.FileHandler(log_path, 'w', 'utf-8')
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except (IOError, OSError) as exc:
        logger.warning(f"Could not create log file: {exc}")
    
    return logger


def check_network(timeout: int = 3) -> bool:
    """
    Verify network connectivity to Modrinth API.
    
    Args:
        timeout: Connection timeout in seconds
        
    Returns:
        True if connection successful, False otherwise
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect(("api.modrinth.com", 443))
        sock.close()
        return True
    except (socket.error, OSError):
        return False


class Application:
    """Main application orchestrator."""
    
    def __init__(
        self,
        config: AppConfig,
        logger: logging.Logger,
        output_dir: Path
    ) -> None:
        self._config = config
        self._logger = logger
        self._output_dir = output_dir
        
        # Initialize dependencies
        self._http_client = StandardHTTPClient(config, logger)
        self._repository = CSVProjectRepository(output_dir, logger)
        self._api = ModrinthAPI(self._http_client, logger)
        self._version_matcher = VersionMatcher()
        self._modpack_extractor = ModpackExtractor(self._http_client, config, logger)
        
        # Initialize services
        self._search_service = SearchService(self._api, logger)
        self._download_service = DownloadService(
            self._api,
            self._http_client,
            self._repository,
            self._version_matcher,
            self._modpack_extractor,
            config,
            logger
        )
    
    def search(
        self,
        query: str,
        loader: Optional[str] = None,
        mc_version: Optional[str] = None,
        project_type: Optional[str] = None,
        limit: int = 10
    ) -> int:
        """Execute search operation."""
        try:
            self._logger.info(f"Searching for: {query}")
            
            filters = {}
            if loader:
                filters['loader'] = loader
            if mc_version:
                filters['version'] = mc_version
            if project_type:
                filters['project_type'] = project_type
            
            results = self._search_service.search(query, filters, limit)
            self._search_service.display_results(results)
            
            return 0
            
        except APIError as exc:
            self._logger.error(f"Search failed: {exc}")
            return 1
        except Exception as exc:
            self._logger.error(f"Unexpected error during search: {exc}", exc_info=True)
            return 1
    
    def download(
        self,
        project_ids: List[str],
        mc_version: str,
        loader: str,
        only_type: Optional[str] = None,
        update: bool = False
    ) -> int:
        """Execute download operation."""
        try:
            self._logger.info(f"Starting download: MC {mc_version} + {loader}")
            self._logger.info(f"Found {len(project_ids)} projects to process")
            
            # Prepare download tasks in parallel
            tasks: List[DownloadTask] = []
            with ThreadPoolExecutor(max_workers=self._config.max_workers) as executor:
                futures = {
                    executor.submit(
                        self._download_service.prepare_download_task,
                        pid, mc_version, loader, self._output_dir, only_type, update
                    ): pid
                    for pid in project_ids
                }
                
                for future in as_completed(futures):
                    pid = futures[future]
                    try:
                        task = future.result()
                        if task:
                            tasks.append(task)
                    except Exception as exc:
                        self._logger.error(f"Failed to process project {pid}: {exc}")
            
            if not tasks:
                self._logger.info("All projects are up to date")
                return 0
            
            self._logger.info(f"Downloading {len(tasks)} projects")
            
            # Execute downloads in parallel
            success_count = 0
            with ThreadPoolExecutor(max_workers=self._config.max_workers) as executor:
                futures = {
                    executor.submit(
                        self._download_service.execute_download,
                        task, self._output_dir
                    ): task
                    for task in tasks
                }
                
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        if future.result():
                            success_count += 1
                    except Exception as exc:
                        self._logger.error(
                            f"Failed to download {task.project_id}: {exc}"
                        )
            
            failed_count = len(tasks) - success_count
            self._logger.info(
                f"Download complete: {success_count} successful, {failed_count} failed"
            )
            
            return 1 if failed_count > 0 else 0
            
        except Exception as exc:
            self._logger.error(f"Download operation failed: {exc}", exc_info=True)
            return 1


# ===========================
# CLI Interface
# ===========================

def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Modrinth Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search for mods
  %(prog)s -s "sodium" -l fabric -v 1.20.1
  
  # Search with filters
  %(prog)s -s "optimization" -l fabric -v 1.20.1 -t mod --limit 20
  
  # Download from collection
  %(prog)s -c collection_id -l fabric -v 1.20.1
  
  # Download specific projects
  %(prog)s -p "sodium,lithium,phosphor" -l fabric -v 1.20.1
  
  # Update existing downloads
  %(prog)s -c collection_id -l fabric -v 1.20.1 -u
  
  # Download to specific directory
  %(prog)s -p "sodium" -l fabric -v 1.20.1 -d ./minecraft/mods

For more information, visit: https://github.com/ownexcept/modrinth-downloader
        """
    )
    
    # Version info
    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {__version__}'
    )
    
    # Search option
    parser.add_argument(
        "-s", "--search",
        metavar="QUERY",
        help="Search for projects (display results only)"
    )
    
    # Download sources
    download_group = parser.add_argument_group('download sources')
    download_group.add_argument(
        "-c", "--collection",
        help="Collection ID to download"
    )
    download_group.add_argument(
        "-p", "--projects",
        help="Comma-separated list of project IDs or slugs"
    )
    
    # Required options (for download mode)
    required_group = parser.add_argument_group('required options')
    required_group.add_argument(
        "-l", "--loader",
        required=True,
        help="Mod loader (fabric/forge/quilt/bukkit/paper/purpur/spigot/etc.)"
    )
    required_group.add_argument(
        "-v", "--mcversion",
        dest="mcversion",
        required=True,
        help="Minecraft version (e.g., 1.20.1, 1.19.4)"
    )
    
    # Output options
    output_group = parser.add_argument_group('output options')
    output_group.add_argument(
        "-d", "--directory",
        type=Path,
        default=Path.cwd(),
        help="Download directory (default: current directory)"
    )
    output_group.add_argument(
        "-ot", "--only-type",
        choices=list(TYPE_FOLDERS.keys()) + ["misc"],
        help="Only download projects of specific type"
    )
    
    # Search filters
    filter_group = parser.add_argument_group('search filters')
    filter_group.add_argument(
        "-t", "--type",
        dest="project_type",
        choices=list(TYPE_FOLDERS.keys()),
        help="Filter by project type (for search)"
    )
    filter_group.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of search results (default: 10)"
    )
    
    # Download options
    download_opts = parser.add_argument_group('download options')
    download_opts.add_argument(
        "-u", "--update",
        action="store_true",
        help="Update existing downloads to latest compatible version"
    )
    download_opts.add_argument(
        "-i", "--ignore-version-ids",
        help="Comma-separated list of version IDs to skip version checks"
    )
    
    # Network options
    network_group = parser.add_argument_group('network options')
    network_group.add_argument(
        "-w", "--workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Number of parallel workers (default: {DEFAULT_MAX_WORKERS})"
    )
    network_group.add_argument(
        "-m", "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Maximum number of retry attempts (default: {DEFAULT_MAX_RETRIES})"
    )
    network_group.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT})"
    )
    network_group.add_argument(
        "-ua", "--user-agent",
        default=DEFAULT_USER_AGENT,
        help=f"Custom User-Agent string (default: {DEFAULT_USER_AGENT})"
    )
    
    # Debug options
    debug_group = parser.add_argument_group('debug options')
    debug_group.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level)"
    )
    
    return parser


def validate_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate command line arguments."""
    # Must have either search OR (collection/projects)
    if not args.search and not args.collection and not args.projects:
        parser.error("Either --search or --collection/--projects must be provided")
    
    # Cannot mix search with download
    if args.search and (args.collection or args.projects):
        parser.error("Cannot use --search with --collection or --projects")
    
    # Validate directory
    if not args.search:
        try:
            args.directory.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as exc:
            parser.error(f"Cannot create directory {args.directory}: {exc}")


def main() -> int:
    """Main entry point with comprehensive error handling."""
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # Validate arguments
        validate_arguments(args, parser)
        
        # Normalize output directory to args.directory/{args.version}-{args.loader}
        args.directory = (args.directory / f"{args.mcversion}-{args.loader}").resolve()
        args.directory.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        log_level = logging.DEBUG if args.verbose else logging.INFO
        logger = setup_logging(args.directory, log_level)
        
        logger.info(f"Modrinth Downloader v{__version__} - Starting")
        
        # Check network connectivity
        if not check_network():
            logger.error("No internet connection detected")
            print("Error: No internet connection. Please check your network.")
            return 1
        
        # Create configuration
        config = AppConfig.from_args(args)
        
        # Initialize application
        app = Application(config, logger, args.directory)
        
        # Execute appropriate operation
        if args.search:
            return app.search(
                query=args.search,
                loader=args.loader,
                mc_version=args.mcversion,
                project_type=args.project_type,
                limit=args.limit
            )
        else:
            # Get project IDs
            if args.collection:
                logger.info(f"Fetching collection: {args.collection}")
                try:
                    api = ModrinthAPI(StandardHTTPClient(config, logger), logger)
                    project_ids = api.get_collection(args.collection)
                except APIError as exc:
                    logger.error(f"Failed to fetch collection: {exc}")
                    print(f"Error: Failed to get collection. Use --projects instead.")
                    return 1
            else:
                project_ids = [p.strip() for p in args.projects.split(',')]
            
            if not project_ids:
                logger.error("No projects found")
                print("Error: No projects to download")
                return 1
            
            return app.download(
                project_ids=project_ids,
                mc_version=args.mcversion,
                loader=args.loader,
                only_type=args.only_type,
                update=args.update
            )
    
    except ConfigurationError as exc:
        print(f"Configuration Error: {exc}")
        return 1
    
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        return 130  # Standard Unix exit code for SIGINT
    
    except ModrinthError as exc:
        print(f"Modrinth Error: {exc}")
        return 1
    
    except Exception as exc:
        print(f"Fatal Error: {exc}")
        if hasattr(args, 'verbose') and args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())