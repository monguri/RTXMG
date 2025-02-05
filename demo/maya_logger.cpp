
#include "./maya_logger.h"

#include <array>

constexpr float3 const blue = { 0.f, 0.f, 1.f };
constexpr float3 const green = { 0.f, 1.f, 0.f };
constexpr float3 const red = { 1.f, 0.f, 0.f };

template <typename VEC3> constexpr char const* float3Format()
{
    static std::array<char const*, 2> const _formats = { "%f %f %f   ", "%lf %lf %lf   ", };
    if constexpr (std::same_as<VEC3, float3>)
        return _formats[0];
    else if constexpr (std::same_as<VEC3, double3>)
        return _formats[1];
    else
    {
        return "unknown type";
        assert(0);
    }
}

template <int ncols = 5> inline void newLine(FILE* m_fp, uint32_t i)
{
    if (i > 0 && (((i + 1) % ncols) == 0))
        std::fprintf(m_fp, "\n\t");
}

template <typename VEC3> static void fillVectorAttr(FILE* m_fp,
    char const* attrName, VEC3 const& value, uint32_t nvalues)
{

    std::fprintf(m_fp, "setAttr \"%s\" -type \"vectorArray\" %d \n\t", attrName, nvalues);
    for (uint32_t i = 0; i < nvalues; ++i)
    {
        std::fprintf(m_fp, float3Format<VEC3>(), value[0], value[1], value[2]);
        newLine(m_fp, i);
    }
    std::fprintf(m_fp, ";\n\n");
}

template <typename VEC3> static void setVectorAttr(FILE* m_fp,
    char const* attrName, std::vector<VEC3> const& values)
{

    uint32_t nvalues = (uint32_t)values.size();
    std::fprintf(m_fp, "setAttr \"%s\" -type \"vectorArray\" %d \n\t", attrName, nvalues);
    for (uint32_t i = 0; i < nvalues; ++i)
    {
        std::fprintf(m_fp, float3Format<VEC3>(), values[i][0], values[i][1], values[i][2]);
        newLine(m_fp, i);
    }
    std::fprintf(m_fp, ";\n\n");
}

//
//
//

void writeHeader(FILE* m_fp, uint32_t mayaVersion = 2023)
{

    std::fprintf(m_fp, "//Maya ASCII %d scene", mayaVersion);
    std::fprintf(m_fp, "requires maya \"2023\";\n");
    std::fprintf(m_fp, "requires -nodeType \"nvCapsuleNode\" \"nvidiaCapsule\" \"1.0.0\";\n");
    std::fprintf(m_fp, "currentUnit -l centimeter -a degree -t film;\n");
}

void writeFooter(FILE* m_fp, char const* m_filepath)
{
    std::fprintf(m_fp, "// End of %s", m_filepath ? m_filepath : "scene file");
}

static void createNode(FILE* m_fp, char const* type, char const* name, char const* parent = nullptr)
{
    std::fprintf(m_fp, "createNode %s -n \"%s\"", type, name);
    if (parent)
        std::fprintf(m_fp, " -p \"%s\"", parent);
    std::fprintf(m_fp, ";\n");
}

static void createTransformNode(FILE* m_fp, char const* name, char const* parent = nullptr)
{
    createNode(m_fp, "transform", name, parent);
}

static void createParticleNode(FILE* m_fp, char const* name, char const* parent, bool streaks = false)
{
    createNode(m_fp, "particle", name, parent);

    if (streaks)
    {
        std::fprintf(m_fp, "setAttr \".particleRenderType\" 6;\n");

        std::fprintf(m_fp, "addAttr -is true -ci true "
            "-sn \"lineWidth\" -ln \"lineWidth\" -dv 1 -min 1 -max 20 -at \"long\";\n");
        std::fprintf(m_fp, "setAttr -k on \".lineWidth\" 2;\n");

        std::fprintf(m_fp, "addAttr -is true -ci true "
            "-sn \"tailFade\" -ln \"tailFade\" -min -1 -max 1 -at \"float\";\n");
        std::fprintf(m_fp, "setAttr \".tailFade\" 1;\n");

        std::fprintf(m_fp, "addAttr -is true -ci true "
            "-sn \"tailSize\" -ln \"tailSize\" -dv 1 -min -100 -max 100 -at \"float\";\n");
        std::fprintf(m_fp, "setAttr \".tailSize\" 1;\n");
    }
}

static void setParticleIDs(FILE* m_fp, uint32_t nparticles)
{
    std::fprintf(m_fp, "setAttr \".id0\" -type \"doubleArray\" %d \n\t", nparticles);
    for (uint32_t i = 0; i < nparticles; ++i)
    {
        std::fprintf(m_fp, "%d ", i);
        newLine<30>(m_fp, i);
    }
    std::fprintf(m_fp, ";\n");
    std::fprintf(m_fp, "setAttr \".nid0\" %d;\n\n", nparticles);
}

static void addParticleColorAttr(FILE* m_fp,
    char const* shapeName, char const* shapePath, float3 const& value, uint32_t nvalues)
{

    std::fprintf(m_fp, "addAttr -s false -ci true -sn \"rgbPP\" -ln \"rgbPP\" -dt \"vectorArray\";\n");
    std::fprintf(m_fp, "addAttr -ci true -h true -sn \"rgbPP0\" -ln \"rgbPP0\" -dt \"vectorArray\";\n");

    fillVectorAttr(m_fp, ".rgbPP0", value, nvalues);

    std::fprintf(m_fp, "connectAttr \"%s|%s.xo[0]\" \"%s|%s.rgbPP\";\n\n",
        shapePath, shapeName, shapePath, shapeName);
}

static void addParticleColorAttr(FILE* m_fp,
    char const* shapeName, char const* shapePath, std::vector<float3> const& values)
{
    std::fprintf(m_fp, "addAttr -s false -ci true -sn \"rgbPP\" -ln \"rgbPP\" -dt \"vectorArray\";\n");
    std::fprintf(m_fp, "addAttr -ci true -h true -sn \"rgbPP0\" -ln \"rgbPP0\" -dt \"vectorArray\";\n");

    setVectorAttr(m_fp, ".rgbPP0", values);

    std::fprintf(m_fp, "connectAttr \"%s|%s.xo[0]\" \"%s|%s.rgbPP\";\n\n",
        shapePath, shapeName, shapePath, shapeName);

}

static void addParticlePositionAttr(FILE* m_fp, std::vector<float3> const& positions)
{
    setVectorAttr(m_fp, ".pos0", positions);
}

//
//
//

std::unique_ptr<MayaLogger> MayaLogger::Create(char const* m_filepath)
{

    if (FILE* m_fp = fopen(m_filepath, "w"))
    {
        writeHeader(m_fp);
        auto logger = std::make_unique<MayaLogger>();
        logger->m_filepath = std::string(m_filepath);
        logger->m_fp = m_fp;
        return logger;
    }
    return nullptr;
}

MayaLogger::~MayaLogger()
{
    writeFooter(m_fp, m_filepath.c_str());
    fclose(m_fp);
}


void MayaLogger::CreateParticles(ParticleDescriptor const& desc)
{

    if (desc.positions.empty())
        return;

    uint32_t nparticles = (uint32_t)desc.positions.size();

    std::string parent = desc.nodePath;

    createTransformNode(m_fp, desc.nodeName.c_str(), parent.c_str());
    std::string shapePath = parent + "|" + desc.nodeName;

    std::string shapeName = desc.nodeName + "_Shape";
    createParticleNode(m_fp, shapeName.c_str(), shapePath.c_str());

    setParticleIDs(m_fp, nparticles);

    addParticlePositionAttr(m_fp, desc.positions);

    if (!desc.colors.empty())
        addParticleColorAttr(m_fp, shapeName.c_str(), shapePath.c_str(), desc.colors);
}
