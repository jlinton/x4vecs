//usr/bin/tail -n +1 $0 | g++ -I/usr/include/SDL2 -O3 -g -lSDL2 pugixml.cpp -msse3 -o ${0%.cpp} -x c++ - && ./${0%.cpp} $1 $2 && rm ./${0%.cpp} ; exit
//
// x4vecs unit test & wireframe renderer.
// Copyright (C) 2013-2015 Jeremy Linton
//
// Build with SDL2-dev installed,
// also put pugixml.[hc]pp, pugiconfig.hpp in the current directory
//
// This started as an alternate unit test for the x4vecs cross platform
// vector/matrix routines I wrote. I actually think its a pretty cool piece of 
// code on its own. It can read the standard COLLADA .dae asset interchange format
// open a graphics window and rotate it using a software wireframe renderer 
//
// The whole thing comes down to a 20 line piece of code that is basically
//
// for (each known polygon)
// {
//   for (each point in the polygon)
//   {
//      SEx4 new_point=trans*Points[Polys[x].points[poly_points]];
//      draw_line_from_last_point_to_this_one();
//   }
// }
//
// All the magic happens in the matrix multiply of `trans*Points[]`
//
// I have another version with a custom line draw routine instead of the 
// SDL2 routine, the idea is that a fast polyfill will provide a gouraud shaded
// routine with just a little more work (and a z-buffer). 
//



#include <stdio.h>

#include <SDL.h>


#define NO_MAIN


#ifdef __ARM_NEON
#define __ARM_NEON__
#endif

#ifdef __ARM_NEON__
#include "neon_objs.cpp"
#endif

#ifdef __SSE4_1__
#include "sse_objs2.cpp"
#endif


#include "pugixml.hpp"

#define    amsk   0xff
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
#define    rmask  0xff000000
#define    gmask  0x00ff0000
#define    bmask  0x0000ff00
#define    amask  0x000000ff
#else
#define    rmask  0x000000ff
#define    gmask  0x0000ff00
#define    bmask  0x00ff0000
#define    amask  0xff000000
#endif

//#define NUMPOINTS 507

#define SHORT_STRING 255
#define MAX_POINTS_PER_POLY 4


// frankly this structure should probably specify the points directly rather than referencing the points array
// the random nature of the point array accesses is going to suck. But this thing is mostly i unittest
struct polylist_t
{
    int num_points;
    int points[MAX_POINTS_PER_POLY];
    int normals[MAX_POINTS_PER_POLY];
};


// this class reads a .DAE file, and encpuslates the 
// the points/normals and polygon list from the .dae file
// it can then be used by UpdateSurface() to draw a wireframe
// representation of the mesh.
//
// In the near future consider creating a couple of meshes and 
// drawing them independently to the canvas
class Mesh
{
  public:
    Mesh(const char *FileName_prm):NUMPOINTS(-1),NUMPOLYS(0),NUMNORMALS(-1),Polys(NULL),Points(NULL),Normals(NULL) { if (ParseCollada(FileName_prm)!=0) throw; }
    ~Mesh(); 
    int ParseCollada(const char *FileName_prm);

    int NUMPOINTS;  //number of points in this mesh
    int NUMPOLYS;   //number of polygons in this mesh
    int NUMNORMALS; //number of normals
    polylist_t *Polys; 

    Vec4 *Points;
    Vec4 *Normals;
  private:
    void PopulateVecs(Vec4 *Vecs,char *SourceStr,float w);
};

Mesh::~Mesh()
{
    if (Points!=NULL)
    {
        delete[] Points;
    }
    if (Normals!=NULL)
    {
        delete[] Normals;
    }
    if (Polys!=NULL)
    {
        delete[] Polys;
    }
}


// This reads the given .DAE file, and treats it like 
// an XML COLLADA file. Pugixml is used to parse the XML
// and the resulting data is pushed into an array of polylist_t
// structures. 
int  Mesh::ParseCollada(const char *FileName_prm)
{
    int ret=0;
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(FileName_prm);

    if (result)
    {
        printf("Parse ok...");
        pugi::xml_node tool = doc.child("COLLADA");
        if (tool)
        {
            printf("1.");
            tool=tool.child("library_geometries");
            if (tool)
            {
                printf("2.");
                tool=tool.child("geometry");
                if (tool)
                {
                    std::string mesh_name=tool.attribute("id").value();
                    printf(" %s 3.",mesh_name.c_str());
                    tool=tool.child("mesh");
                    if (tool)
                    {
                        printf("4.");
                        //tool=tool.child("source");
                        pugi::xml_node positions=tool.find_child_by_attribute("source","id",(mesh_name+"-positions").c_str());
                        if (positions)
                        {
                            printf("5.");
                            positions=positions.child("float_array");
                            if (positions)
                            {
                                printf("Float id=%s .",positions.attribute("id").value());
                                printf("total points %s .",positions.attribute("count").value());

                                NUMPOINTS=positions.attribute("count").as_int()/3;
                                Points=new Vec4[NUMPOINTS+5];
                                //printf("points %s",positions.first_child().value());
                                PopulateVecs(Points,(char *)positions.first_child().value(),1);

                                //          exit(1);
                            }
                        }
                        
                        pugi::xml_node normals=tool.find_child_by_attribute("source","id",(mesh_name+"-normals").c_str());                      
                        if (normals)
                        {
                            printf("6.");
                            normals=normals.child("float_array");
                            if (normals)
                            {
                                NUMNORMALS=normals.attribute("count").as_int()/3;
                                printf(" total normals=%d .",NUMNORMALS);
                                Normals=new Vec4[NUMNORMALS+5];

                                PopulateVecs(Normals,(char *)normals.first_child().value(),0);
                            }
                        }

                        Polys=new polylist_t[NUMNORMALS]; // use the normals to approximate
                            

                        pugi::xml_node polylist=tool.child("polylist");
                        int iter=0;
                        while (polylist)
                        {
                            NUMPOLYS+=polylist.attribute("count").as_int();
                            printf(" polycount=%d ",NUMPOLYS);
                            
                            if (NUMPOLYS>0)
                            {
                                printf("7.");
                                // TODO, verify polylist count=NUMPOLYS
                                pugi::xml_node vcount=polylist.child("vcount");
                                pugi::xml_node pl=polylist.child("p");

                                pugi::xml_node textcords=polylist.find_child_by_attribute("input","semantic","TEXCOORD");

                                if (textcords)
                                {
                                    printf("Skipping textcords ");
                                }

                                if (pl)
                                {
                                    printf("8.");
                                    char *hold_vtk;
                                    char *hold_ptk;
                                
                                    char *vtk=strtok_r((char *)vcount.first_child().value()," ",&hold_vtk);
                                    char *ptk=strtok_r((char *)pl.first_child().value()," ",&hold_ptk);
                                    do 
                                    {
                                        int num_verts;
                                        sscanf(vtk,"%d",&num_verts);
                                        Polys[iter].num_points=num_verts;
                                        for (int cv=0;cv<num_verts;cv++)
                                        {
                                            sscanf(ptk,"%d",&Polys[iter].points[cv]);
                                            ptk=strtok_r(NULL," ",&hold_ptk);
                                            sscanf(ptk,"%d",&Polys[iter].normals[cv]);
                                            ptk=strtok_r(NULL," ",&hold_ptk);
                                            if (textcords)
                                            {
                                                // TEXCOORD
                                                int tx;
                                                sscanf(ptk,"%d",&tx);
                                                ptk=strtok_r(NULL," ",&hold_ptk);
                                            }
                                        }
                                        iter++;
                                        vtk=strtok_r(NULL," ",&hold_vtk);
                                    } while (vtk!=NULL);
                                

                                }
                            }
                            polylist=polylist.next_sibling("polylist");
                        } 


                    }
                }
            }
        }
        printf("Done\n");fflush(0);
        //
        // Debug the mesh parsing.
        //
        //std::cout << "XML [" << source << "] parsed without errors, attr value: [" << doc.child("node").attribute("attr").value() << "]\n\n";
        //for (child("library_geometries").child("geometry").child("mesh").child("source_id").child("float_array"); tool; tool = tool.next_sibling("float_array"))
        //{
            //std::cout << "Tool " << tool.attribute("Filename").value();
            //std::cout << ": AllowRemote " << tool.attribute("AllowRemote").as_bool();
            //std::cout << ", Timeout " << tool.attribute("Timeout").as_int();
            //std::cout << ", Description '" << tool.child_value("Description") << "'\n";
    
        //}
    }
    else
    {
        printf("Parse error '%s' at line:%d\n",result.description(),result.offset);
        ret=-1;
        //std::cout << "XML [" << source << "] parsed with errors, attr value: [" << doc.child("node").attribute("attr").value() << "]\n";
        //std::cout << "Error description: " << result.description() << "\n";
        //std::cout << "Error offset: " << result.offset << " (error at [..." << (source + result.offset) << "]\n\n";
    }
    return ret;
}

// Given a XML parsed out string from the .dae file, turn it into a set of 
// 1x4 vectors.
//
// bad mojo here, we are casting the const away, and modifying the string!
// works for now because its pugixml, but in the future we probably should use strtok_r
void Mesh::PopulateVecs(Vec4 *Vecs,char *SourceStr,float w)
{
    char *tk=strtok(SourceStr," ");
    int iter=0;
    do 
    {
        // pick off a single x,y,z set of points from the given source string
        // uses 'w' as the 4th part of the vector, which indicates if we are talking
        // about a point or a vector itself.
        float x,y,z;
        sscanf(tk,"%f",&x);
        tk=strtok(NULL," ");
        sscanf(tk,"%f",&y);
        tk=strtok(NULL," ");
        sscanf(tk,"%f",&z);
        tk=strtok(NULL," ");
        Vecs[iter] = Vec4(x, y, z ,w);
        iter++;
    } while (tk!=NULL);
}


// This is the renderer
// it spins the given mesh 
// and plots it to the SDL renderer
//
// Draw wireframe rendering of mesh_data to current rendering surface,
// scaled by width, height, (also implicitly transformed to +200,+200)
int UpdateSurface(SDL_Renderer *Rndr,Mesh *mesh_data,int Width,int Height,int color,float scale)
{

    SDL_SetRenderDrawColor(Rndr,0,0,0,amsk);
    SDL_RenderClear(Rndr);

    Scale scll(scale,scale,scale);  

    RotateXRad rot(float(color)/10.0);
//  RotateYRad rot2(float(color)/10.0);
    RotateYRad rot2(2.0);
    Projection pro(Width,Height);
    SSEx4Matrix trans=pro*rot*rot2*scll;

    SDL_SetRenderDrawColor(Rndr,color,0,0,amsk);
                
    // for each polygon in mesh
    for (int x=0;x<mesh_data->NUMPOLYS;x++)
    {
        int prev_x ,prev_y;
        int first_x, first_y;

        prev_x = prev_y = first_x = first_y = 0; // fix gcc warning

        // for each point in the polygon
        for (int poly_points=0;poly_points<mesh_data->Polys[x].num_points;poly_points++)
        {
            // ok pick out the point for this polygon and transform it
            Vec4 new_point=trans*mesh_data->Points[mesh_data->Polys[x].points[poly_points]];

            //extract the x and y from the vector
            int x=new_point[0]; 
            int y=new_point[1]; 

            x+=200; //shift the rotation to somewhere visible on the canvas
            y+=200; 

            // start drawing lines when we have transformed at least two points
            if (poly_points!=0)
            {
                //printf("Draw line %d:%d to %d:%d\n",prev_x,prev_y,x,y);
                SDL_RenderDrawLine(Rndr,prev_x,prev_y,x,y);
            }
            else
            {
                first_x=x;
                first_y=y;
            }

            prev_x=x;
            prev_y=y;       

        }
        SDL_RenderDrawLine(Rndr,prev_x,prev_y,first_x,first_y);
    }
    return 0;
}

#define TOTAL_ROTATIONS 10

#include <unistd.h>
int main(int argc, char **argv)
{
    if (argc<2)
    {
        printf("Use like %s filename.dae\n",argv[0]);
        return -1;
    }

    //read and construct a mesh from the given .dae file
    Mesh mesh_from_file(argv[1]); 

    float scale=80; //use 80 for monkey.dae and 0.2 for heinkel.dae 
    if (argc>2)
    {
        sscanf(argv[2],"%f",&scale);
        printf("Scale set to %f\n",scale);
    }

    // initialize the SDL video surface
    if (SDL_Init(SDL_INIT_VIDEO)!=-1)
    {
        SDL_Window *win = SDL_CreateWindow("Hello World!", 100, 100, 640, 480, SDL_WINDOW_SHOWN);
        if (win!=NULL)
        {
            SDL_Renderer *render=SDL_CreateRenderer(win,-1,SDL_RENDERER_ACCELERATED|SDL_RENDERER_PRESENTVSYNC);
            //SDL_Renderer *render=SDL_CreateRenderer(win,-1,SDL_RENDERER_ACCELERATED);
            if (render!=NULL)
            {
                SDL_Surface *surf=SDL_CreateRGBSurface(0,640,480,32,rmask,gmask,bmask,amask);
                if (surf)
                {   
                    SDL_Texture *tex = SDL_CreateTextureFromSurface(render, surf);
                    if (tex)
                    {
                        //
                        // Ok, all the SDL setup worked, start spinning the mesh
                        //
                        for (int iteration=0;iteration<TOTAL_ROTATIONS;iteration++)
                        {
                            for (int color=0;color<255;color+=2)
                            {
                                SDL_Event event;
                                if (SDL_PollEvent(&event))
                                {
                                    if (event.type==SDL_QUIT)
                                    {
                                        iteration=TOTAL_ROTATIONS;
                                        break;
                                    }
                                }
                                UpdateSurface(render,&mesh_from_file,640,480,color,scale);
                                
                                SDL_RenderPresent(render);
                            }
                        }
                        //
                        // Done spinning the mesh, lets cleanup
                        //
                        SDL_DestroyTexture(tex);
                    }                   
                    SDL_FreeSurface(surf);
                }
                SDL_DestroyRenderer(render);
            }
            SDL_DestroyWindow(win);
        }
        SDL_Quit();
    }
	return 0;
}
